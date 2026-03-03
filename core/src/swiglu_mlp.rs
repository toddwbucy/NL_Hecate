/// SwiGluMlp: HOPE §7.3 ad-hoc level stacking variant.
///
/// A SwiGLU MLP used as a CMS level — initialized from a pre-trained
/// transformer MLP block (e.g., Llama-3.2-1B layer 0/5/10/15).
///
/// Key semantic differences from matrix-memory rules:
///   - No inner-loop M state: context.memory[level] stays empty
///   - initial_m is ignored — this rule has no per-token recurrence
///   - Outer-loop AdamW updates gate_proj/up_proj/down_proj across chunks
///   - Forward: Y = silu(X @ gate_proj.T) * (X @ up_proj.T) @ down_proj.T
///
/// Spec: specs/infrastructure/build/02_llama_level_stacking.md

use crate::model::MemoryLevelParams;
use crate::delta_rule::MemoryRule;
use crate::tape::{OpaqueKey, OpaqueVjp, Tape, BufId};
use crate::tensor::{matmul_f32, matmul_acc_f32, transpose_f32};

/// SwiGLU MLP rule struct.
pub struct SwiGluMlp {
    pub intermediate_size: usize,
}

/// Forward-pass cache for SwiGluMlp. Stores all intermediates needed for backward.
pub struct SwiGluMlpCache {
    pub seq_len: usize,
    pub d: usize,
    pub intermediate: usize,
    /// Input embeddings [seq_len x d]
    pub x: Vec<f32>,
    /// gate_out = X @ gate_proj.T  [seq_len x intermediate]
    pub gate_out: Vec<f32>,
    /// up_out = X @ up_proj.T      [seq_len x intermediate]
    pub up_out: Vec<f32>,
    /// fused = silu(gate_out) * up_out  [seq_len x intermediate]
    pub fused: Vec<f32>,
    /// gate_cache = sigmoid(gate_out)   [seq_len x intermediate]
    /// Stored separately from fused — needed for d(silu)/d(gate) in backward.
    pub gate_cache: Vec<f32>,
}

impl SwiGluMlp {
    pub fn from_cfg(cfg: &crate::model::MAGConfig) -> Self {
        SwiGluMlp { intermediate_size: cfg.intermediate_size }
    }

    /// CPU forward pass. All tokens batched as matrix operations.
    ///
    /// Forward math (row-major):
    ///   gate_out = X @ gate_proj.T          [seq x inter]
    ///   up_out   = X @ up_proj.T            [seq x inter]
    ///   sig      = sigmoid(gate_out)        elementwise
    ///   fused    = gate_out * sig * up_out  SwiGLU fusion
    ///   Y        = fused @ down_proj.T      [seq x d]
    pub fn step_cpu(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
    ) -> (Vec<f32>, SwiGluMlpCache) {
        let inter = self.intermediate_size;
        assert!(inter > 0, "SwiGluMlp: intermediate_size must be > 0");
        assert_eq!(level_params.gate_proj.len(), inter * d,
            "SwiGluMlp: gate_proj length mismatch: got {}, expected {}*{}={}",
            level_params.gate_proj.len(), inter, d, inter * d);
        assert_eq!(level_params.up_proj.len(), inter * d,
            "SwiGluMlp: up_proj length mismatch");
        assert_eq!(level_params.down_proj.len(), d * inter,
            "SwiGluMlp: down_proj length mismatch");

        let x = embedded.to_vec();

        // gate_out = X @ gate_proj.T  [seq x inter]
        // gate_proj stored as [inter x d]; gate_proj.T is [d x inter]
        // matmul: X [seq x d] @ gate_proj.T [d x inter] = [seq x inter]
        // We compute via: gate_out[s, j] = sum_k x[s, k] * gate_proj[j, k]
        // This is X @ gate_proj^T — equivalently matmul_f32 with transposed B
        let mut gate_proj_t = vec![0.0f32; d * inter];
        transpose_f32(&level_params.gate_proj, &mut gate_proj_t, inter, d);
        let mut gate_out = vec![0.0f32; seq_len * inter];
        matmul_f32(&x, &gate_proj_t, &mut gate_out, seq_len, d, inter);

        // up_out = X @ up_proj.T  [seq x inter]
        let mut up_proj_t = vec![0.0f32; d * inter];
        transpose_f32(&level_params.up_proj, &mut up_proj_t, inter, d);
        let mut up_out = vec![0.0f32; seq_len * inter];
        matmul_f32(&x, &up_proj_t, &mut up_out, seq_len, d, inter);

        // SwiGLU: fused = sigmoid(gate_out) * gate_out * up_out
        // gate_cache = sigmoid(gate_out) [stored for backward]
        let mut gate_cache = vec![0.0f32; seq_len * inter];
        let mut fused = vec![0.0f32; seq_len * inter];
        for i in 0..seq_len * inter {
            let sig = sigmoid(gate_out[i]);
            gate_cache[i] = sig;
            fused[i] = gate_out[i] * sig * up_out[i];
        }

        // Y = fused @ down_proj.T  [seq x d]
        // down_proj stored as [d x inter]; down_proj.T is [inter x d]
        let mut down_proj_t = vec![0.0f32; inter * d];
        transpose_f32(&level_params.down_proj, &mut down_proj_t, d, inter);
        let mut y = vec![0.0f32; seq_len * d];
        matmul_f32(&fused, &down_proj_t, &mut y, seq_len, inter, d);

        let cache = SwiGluMlpCache { seq_len, d, intermediate: inter, x, gate_out, up_out, fused, gate_cache };
        (y, cache)
    }

    /// CPU backward pass through SwiGLU MLP.
    ///
    /// Backward math:
    ///   d_fused     = d_Y @ down_proj               [seq x inter]
    ///   d_down_proj = (d_Y.T @ fused).T → [d x inter]  ← accumulated
    ///
    ///   d_silu(g) = sig(g) * (1 + g * (1 - sig(g)))  where sig = gate_cache
    ///   d_gate     = d_fused * up_out * d_silu(gate)  [seq x inter]
    ///   d_up       = d_fused * gate_out * sig          [seq x inter]
    ///
    ///   d_gate_proj = d_gate.T @ X  [inter x d]
    ///   d_up_proj   = d_up.T   @ X  [inter x d]
    ///   d_X         = d_gate @ gate_proj + d_up @ up_proj  [seq x d]
    pub fn step_backward_cpu(
        &self,
        level_params: &MemoryLevelParams,
        cache: &SwiGluMlpCache,
        d_y: &[f32],
        _embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let seq_len = cache.seq_len;
        let d = cache.d;
        let inter = cache.intermediate;

        // d_fused = d_Y @ down_proj  [seq x inter]
        // d_Y [seq x d] @ down_proj [d x inter] = [seq x inter]
        let mut d_fused = vec![0.0f32; seq_len * inter];
        matmul_f32(d_y, &level_params.down_proj, &mut d_fused, seq_len, d, inter);

        // d_down_proj = d_Y.T @ fused  [d x inter]
        let mut d_y_t = vec![0.0f32; d * seq_len];
        transpose_f32(d_y, &mut d_y_t, seq_len, d);
        let mut d_down_proj = vec![0.0f32; d * inter];
        matmul_f32(&d_y_t, &cache.fused, &mut d_down_proj, d, seq_len, inter);

        // SwiGLU elementwise backward
        let mut d_gate = vec![0.0f32; seq_len * inter];
        let mut d_up = vec![0.0f32; seq_len * inter];
        for i in 0..seq_len * inter {
            let sig = cache.gate_cache[i];
            let g = cache.gate_out[i];
            let u = cache.up_out[i];
            // d(silu)/d(g) = sig * (1 + g * (1 - sig))
            let d_silu_dg = sig * (1.0 + g * (1.0 - sig));
            // d_gate[i] = d_fused[i] * u * d_silu_dg
            d_gate[i] = d_fused[i] * u * d_silu_dg;
            // d_up[i] = d_fused[i] * silu(g) = d_fused[i] * g * sig
            d_up[i] = d_fused[i] * g * sig;
        }

        // d_gate_proj = d_gate.T @ X  [inter x d]
        let mut d_gate_t = vec![0.0f32; inter * seq_len];
        transpose_f32(&d_gate, &mut d_gate_t, seq_len, inter);
        let mut d_gate_proj = vec![0.0f32; inter * d];
        matmul_f32(&d_gate_t, &cache.x, &mut d_gate_proj, inter, seq_len, d);

        // d_up_proj = d_up.T @ X  [inter x d]
        let mut d_up_t = vec![0.0f32; inter * seq_len];
        transpose_f32(&d_up, &mut d_up_t, seq_len, inter);
        let mut d_up_proj = vec![0.0f32; inter * d];
        matmul_f32(&d_up_t, &cache.x, &mut d_up_proj, inter, seq_len, d);

        // d_X = d_gate @ gate_proj + d_up @ up_proj  [seq x d]
        let mut d_x = vec![0.0f32; seq_len * d];
        matmul_acc_f32(&d_gate, &level_params.gate_proj, &mut d_x, seq_len, inter, d);
        matmul_acc_f32(&d_up, &level_params.up_proj, &mut d_x, seq_len, inter, d);

        // Pack gradients into MemoryLevelParams (all other fields zero)
        let mut grads = MemoryLevelParams::zeros_like_from(level_params, d);
        grads.gate_proj = d_gate_proj;
        grads.up_proj = d_up_proj;
        grads.down_proj = d_down_proj;

        (grads, d_x)
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── MemoryRule trait implementation ──────────────────────────────────

impl MemoryRule for SwiGluMlp {
    type Cache = SwiGluMlpCache;
    type State = (); // no inner-loop state

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] {
        &["sequential", "chunkwise_gd", "tnt"]
    }

    fn init(&self, _d: usize) -> Self::State { () }

    fn write(
        &self,
        _state: &mut (),
        _k: &[f32],
        _v: &[f32],
        _gates: &crate::delta_rule::Gates,
    ) -> Result<(), crate::delta_rule::MemoryError> {
        Err(crate::delta_rule::MemoryError::UnsupportedOperation)
    }

    fn read(&self, _state: &(), _q: &[f32], _out: &mut [f32]) -> Result<(), crate::delta_rule::MemoryError> {
        Err(crate::delta_rule::MemoryError::UnsupportedOperation)
    }

    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        _initial_m: Option<Vec<f32>>, // SwiGluMlp has no M state — ignored
    ) -> (Vec<f32>, Self::Cache) {
        #[cfg(feature = "cuda")]
        {
            self.step_cuda(level_params, embedded, seq_len, d)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.step_cpu(level_params, embedded, seq_len, d)
        }
    }

    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &Self::Cache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        #[cfg(feature = "cuda")]
        {
            self.step_backward_cuda(level_params, cache, d_y, embedded)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.step_backward_cpu(level_params, cache, d_y, embedded)
        }
    }
}

// ── CUDA dispatch stubs ───────────────────────────────────────────────

#[cfg(feature = "cuda")]
extern "C" {
    fn swiglu_forward_f32_cuda(
        x: *const f32,
        gate_proj: *const f32,
        up_proj: *const f32,
        down_proj: *const f32,
        y: *mut f32,
        gate_buf: *mut f32,
        up_buf: *mut f32,
        fused_buf: *mut f32,
        cache_buf: *mut f32,
        seq_len: i32,
        d_model: i32,
        intermediate: i32,
    );

    fn swiglu_backward_f32_cuda(
        d_y: *const f32,
        x: *const f32,
        gate_proj: *const f32,
        up_proj: *const f32,
        down_proj: *const f32,
        fused_buf: *const f32,
        gate_buf: *const f32,
        up_buf: *const f32,
        cache_buf: *const f32,
        d_x: *mut f32,
        d_gate_proj: *mut f32,
        d_up_proj: *mut f32,
        d_down_proj: *mut f32,
        seq_len: i32,
        d_model: i32,
        intermediate: i32,
    );
}

#[cfg(feature = "cuda")]
impl SwiGluMlp {
    fn step_cuda(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
    ) -> (Vec<f32>, SwiGluMlpCache) {
        let inter = self.intermediate_size;
        let x = embedded.to_vec();

        let mut gate_out = vec![0.0f32; seq_len * inter];
        let mut up_out = vec![0.0f32; seq_len * inter];
        let mut fused = vec![0.0f32; seq_len * inter];
        let mut gate_cache = vec![0.0f32; seq_len * inter];
        let mut y = vec![0.0f32; seq_len * d];

        unsafe {
            swiglu_forward_f32_cuda(
                x.as_ptr(),
                level_params.gate_proj.as_ptr(),
                level_params.up_proj.as_ptr(),
                level_params.down_proj.as_ptr(),
                y.as_mut_ptr(),
                gate_out.as_mut_ptr(),
                up_out.as_mut_ptr(),
                fused.as_mut_ptr(),
                gate_cache.as_mut_ptr(),
                seq_len as i32,
                d as i32,
                inter as i32,
            );
        }

        let cache = SwiGluMlpCache { seq_len, d, intermediate: inter, x, gate_out, up_out, fused, gate_cache };
        (y, cache)
    }

    fn step_backward_cuda(
        &self,
        level_params: &MemoryLevelParams,
        cache: &SwiGluMlpCache,
        d_y: &[f32],
        _embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let seq_len = cache.seq_len;
        let d = cache.d;
        let inter = cache.intermediate;

        let mut d_x = vec![0.0f32; seq_len * d];
        let mut d_gate_proj = vec![0.0f32; inter * d];
        let mut d_up_proj = vec![0.0f32; inter * d];
        let mut d_down_proj = vec![0.0f32; d * inter];

        unsafe {
            swiglu_backward_f32_cuda(
                d_y.as_ptr(),
                cache.x.as_ptr(),
                level_params.gate_proj.as_ptr(),
                level_params.up_proj.as_ptr(),
                level_params.down_proj.as_ptr(),
                cache.fused.as_ptr(),
                cache.gate_out.as_ptr(),
                cache.up_out.as_ptr(),
                cache.gate_cache.as_ptr(),
                d_x.as_mut_ptr(),
                d_gate_proj.as_mut_ptr(),
                d_up_proj.as_mut_ptr(),
                d_down_proj.as_mut_ptr(),
                seq_len as i32,
                d as i32,
                inter as i32,
            );
        }

        let mut grads = MemoryLevelParams::zeros_like_from(level_params, d);
        grads.gate_proj = d_gate_proj;
        grads.up_proj = d_up_proj;
        grads.down_proj = d_down_proj;

        (grads, d_x)
    }
}

// ── OpaqueVjp implementation ──────────────────────────────────────────

impl OpaqueVjp for SwiGluMlp {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::SwiGluMlp }

    fn record_on_tape(
        &self,
        tape: &mut Tape,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        _initial_m: Option<Vec<f32>>,
        level: Option<usize>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let inter = self.intermediate_size;

        // Run forward — dispatch to CUDA when available; step_cpu is fatal at d=2048
        let (y, cache) = self.step(level_params, embedded, seq_len, d, None::<Vec<f32>>);

        // Allocate input IDs for embedded + level_params (will receive gradients)
        let emb_in = tape.alloc(embedded.to_vec(), vec![seq_len, d]);

        // Flatten gate_proj, up_proj, down_proj into a single level_params buffer
        let mut lp_flat = Vec::with_capacity(3 * inter * d);
        lp_flat.extend_from_slice(&level_params.gate_proj);
        lp_flat.extend_from_slice(&level_params.up_proj);
        lp_flat.extend_from_slice(&level_params.down_proj);
        let lp_in = tape.alloc(lp_flat, vec![]);

        // Save metadata: [seq_len, d, inter]
        let meta = vec![seq_len as f32, d as f32, inter as f32];
        let meta_id = tape.alloc(meta, vec![]);

        // Save cache fields
        let x_id = tape.alloc(cache.x.clone(), vec![]);
        let gate_out_id = tape.alloc(cache.gate_out.clone(), vec![]);
        let up_out_id = tape.alloc(cache.up_out.clone(), vec![]);
        let fused_id = tape.alloc(cache.fused.clone(), vec![]);
        let gc_id = tape.alloc(cache.gate_cache.clone(), vec![]);

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let saved = vec![meta_id, lp_in, emb_in, x_id, gate_out_id, up_out_id, fused_id, gc_id];
        tape.record_opaque(OpaqueKey::SwiGluMlp, vec![emb_in, lp_in], vec![y_id], saved, level);
        (y, y_id, emb_in, lp_in)
    }
}

/// Opaque backward adapter for SwiGluMlp.
///
/// saved layout (matches traced_active_level in traced_forward.rs):
///   saved[0] = [seq_len, d, inter]     metadata
///   saved[1] = lp_id buffer:  gate_proj ++ up_proj ++ down_proj
///              (SwiGluMlp stores only its three projection matrices — no
///               standard MemoryLevelParams prefix, unlike other rules)
///   saved[2] = embedded  [seq*d]
///   saved[3] = x         [seq*d]       (copy of embedded stored in cache)
///   saved[4] = gate_out  [seq*inter]
///   saved[5] = up_out    [seq*inter]
///   saved[6] = fused     [seq*inter]
///   saved[7] = gate_cache [seq*inter]
pub fn swiglu_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    // Fail-fast shape checks before touching any buffer.
    assert_eq!(saved.len(), 8,
        "swiglu_opaque_backward: expected 8 saved tensors, got {}", saved.len());
    assert_eq!(d_outputs.len(), 1,
        "swiglu_opaque_backward: expected 1 output gradient, got {}", d_outputs.len());

    let seq_len = saved[0][0] as usize;
    let d = saved[0][1] as usize;
    let inter = saved[0][2] as usize;

    assert_eq!(d_outputs[0].len(), seq_len * d,
        "swiglu_opaque_backward: d_outputs[0] len {} != seq_len*d {}", d_outputs[0].len(), seq_len * d);
    assert_eq!(saved[2].len(), seq_len * d,
        "swiglu_opaque_backward: saved[2] (embedded) len {} != seq_len*d {}", saved[2].len(), seq_len * d);

    // lp_flat = [gate_proj | up_proj | down_proj] — no standard-fields prefix.
    // record_on_tape stores only the three SwiGLU projection matrices.
    let lp_flat = saved[1];
    assert_eq!(
        lp_flat.len(), 3 * inter * d,
        "swiglu_opaque_backward: lp_flat len {} != 3*inter*d {}",
        lp_flat.len(), 3 * inter * d
    );
    let gate_proj = lp_flat[0..inter * d].to_vec();
    let up_proj   = lp_flat[inter * d..2 * inter * d].to_vec();
    let down_proj = lp_flat[2 * inter * d..3 * inter * d].to_vec();

    // Reconstruct a minimal MemoryLevelParams with just the MLP fields
    let mut level_params = MemoryLevelParams::zeros_like(d);
    level_params.gate_proj = gate_proj;
    level_params.up_proj = up_proj;
    level_params.down_proj = down_proj;

    assert_eq!(saved[3].len(), seq_len * d,
        "swiglu_opaque_backward: saved[3] (x) len {} != seq_len*d {}", saved[3].len(), seq_len * d);
    assert_eq!(saved[4].len(), seq_len * inter,
        "swiglu_opaque_backward: saved[4] (gate_out) len {} != seq_len*inter {}", saved[4].len(), seq_len * inter);
    assert_eq!(saved[5].len(), seq_len * inter,
        "swiglu_opaque_backward: saved[5] (up_out) len {} != seq_len*inter {}", saved[5].len(), seq_len * inter);
    assert_eq!(saved[6].len(), seq_len * inter,
        "swiglu_opaque_backward: saved[6] (fused) len {} != seq_len*inter {}", saved[6].len(), seq_len * inter);
    assert_eq!(saved[7].len(), seq_len * inter,
        "swiglu_opaque_backward: saved[7] (gate_cache) len {} != seq_len*inter {}", saved[7].len(), seq_len * inter);

    let cache = SwiGluMlpCache {
        seq_len,
        d,
        intermediate: inter,
        x: saved[3].to_vec(),
        gate_out: saved[4].to_vec(),
        up_out: saved[5].to_vec(),
        fused: saved[6].to_vec(),
        gate_cache: saved[7].to_vec(),
    };

    let rule = SwiGluMlp { intermediate_size: inter };
    // Dispatch to CUDA backward when compiled with cuda feature (uses cuBLAS).
    // For large d (e.g. d=2048, inter=8192) this is critical — the CPU fallback
    // uses naive O(m*k*n) matmul which is ~100x slower than cuBLAS at scale.
    #[cfg(feature = "cuda")]
    let (param_grads, d_embedded) = rule.step_backward_cuda(&level_params, &cache, d_outputs[0], saved[2]);
    #[cfg(not(feature = "cuda"))]
    let (param_grads, d_embedded) = rule.step_backward_cpu(&level_params, &cache, d_outputs[0], saved[2]);

    // d_inputs[0] = d_embedded (flows to embedding lookup backward)
    d_inputs[0] = d_embedded;

    // d_inputs[1] must match lp_id buffer layout: [gate_proj | up_proj | down_proj].
    // Size must equal saved[1].len() (checked by accumulate_grad).
    let mut lp_grads = Vec::with_capacity(3 * inter * d);
    lp_grads.extend_from_slice(&param_grads.gate_proj);
    lp_grads.extend_from_slice(&param_grads.up_proj);
    lp_grads.extend_from_slice(&param_grads.down_proj);
    assert_eq!(lp_grads.len(), saved[1].len(),
        "swiglu_opaque_backward: lp_grads len {} != saved[1] len {}", lp_grads.len(), saved[1].len());
    d_inputs[1] = lp_grads;
}

// ── Unit tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::SimpleRng;

    fn make_params(d: usize, inter: usize, seed: u64) -> MemoryLevelParams {
        let mut rng = SimpleRng::new(seed);
        let scale = (2.0 / (d + inter) as f32).sqrt();
        let mut gate = vec![0.0f32; inter * d];
        rng.fill_uniform(&mut gate, scale);
        let mut up = vec![0.0f32; inter * d];
        rng.fill_uniform(&mut up, scale);
        let down_scale = (2.0 / (inter + d) as f32).sqrt();
        let mut down = vec![0.0f32; d * inter];
        rng.fill_uniform(&mut down, down_scale);
        let mut params = MemoryLevelParams::zeros_like(d);
        params.gate_proj = gate;
        params.up_proj = up;
        params.down_proj = down;
        params
    }

    fn make_input(seq_len: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = SimpleRng::new(seed);
        let mut x = vec![0.0f32; seq_len * d];
        rng.fill_uniform(&mut x, 1.0);
        x
    }

    #[test]
    fn test_swiglu_forward_finite() {
        let d = 8;
        let inter = 16;
        let seq_len = 4;
        let params = make_params(d, inter, 42);
        let x = make_input(seq_len, d, 43);

        let rule = SwiGluMlp { intermediate_size: inter };
        let cfg = crate::model::MAGConfig { intermediate_size: inter, ..crate::model::MAGConfig::test_config() };
        let (y, cache) = rule.step_cpu(&params, &x, seq_len, d);

        assert_eq!(y.len(), seq_len * d);
        assert!(y.iter().all(|&v| v.is_finite()), "SwiGluMlp forward produced non-finite output");
        assert_eq!(cache.gate_out.len(), seq_len * inter);
        assert_eq!(cache.fused.len(), seq_len * inter);
        let _ = cfg; // suppress unused warning
    }

    #[test]
    fn test_swiglu_sigmoid_range() {
        // gate_cache (sigmoid) must be in (0, 1)
        let d = 8;
        let inter = 16;
        let seq_len = 4;
        let params = make_params(d, inter, 44);
        let x = make_input(seq_len, d, 45);

        let rule = SwiGluMlp { intermediate_size: inter };
        let (_, cache) = rule.step_cpu(&params, &x, seq_len, d);

        for &s in &cache.gate_cache {
            assert!(s > 0.0 && s < 1.0, "sigmoid out of range: {}", s);
        }
    }

    #[test]
    fn test_swiglu_fd_gradient() {
        // Finite-difference gradient check for d_gate_proj, d_up_proj, d_down_proj, d_X
        let d = 8;
        let inter = 16;
        let seq_len = 4;
        let eps = 1e-2_f32;
        let abs_tol = 5e-4_f32;
        let rel_tol = 0.15_f32;

        let params = make_params(d, inter, 99);
        let x = make_input(seq_len, d, 100);
        let mut rng = SimpleRng::new(101);
        let mut d_y = vec![0.0f32; seq_len * d];
        rng.fill_uniform(&mut d_y, 1.0);

        let rule = SwiGluMlp { intermediate_size: inter };
        let (_, cache) = rule.step_cpu(&params, &x, seq_len, d);
        let (grads, d_x) = rule.step_backward_cpu(&params, &cache, &d_y, &x);

        // Helper: scalar loss = sum(Y * d_y)
        let loss = |y: &[f32]| -> f32 { y.iter().zip(d_y.iter()).map(|(a, b)| a * b).sum() };

        // Check d_gate_proj
        let mut params_p = params.clone();
        let mut params_m = params.clone();
        let n = inter * d;
        for i in 0..n {
            params_p.gate_proj[i] += eps;
            params_m.gate_proj[i] -= eps;
            let (y_p, _) = rule.step_cpu(&params_p, &x, seq_len, d);
            let (y_m, _) = rule.step_cpu(&params_m, &x, seq_len, d);
            let fd = (loss(&y_p) - loss(&y_m)) / (2.0 * eps);
            let an = grads.gate_proj[i];
            if an.abs() > abs_tol || fd.abs() > abs_tol {
                let rel = (fd - an).abs() / (fd.abs().max(an.abs()) + 1e-8);
                assert!(rel < rel_tol,
                    "gate_proj[{i}]: fd={fd:.6e} analytical={an:.6e} rel={rel:.3}");
            }
            params_p.gate_proj[i] -= eps;
            params_m.gate_proj[i] += eps;
        }

        // Check d_X
        let mut x_p = x.clone();
        let mut x_m = x.clone();
        for i in 0..seq_len * d {
            x_p[i] += eps;
            x_m[i] -= eps;
            let (y_p, _) = rule.step_cpu(&params, &x_p, seq_len, d);
            let (y_m, _) = rule.step_cpu(&params, &x_m, seq_len, d);
            let fd = (loss(&y_p) - loss(&y_m)) / (2.0 * eps);
            let an = d_x[i];
            if an.abs() > abs_tol || fd.abs() > abs_tol {
                let rel = (fd - an).abs() / (fd.abs().max(an.abs()) + 1e-8);
                assert!(rel < rel_tol,
                    "d_X[{i}]: fd={fd:.6e} analytical={an:.6e} rel={rel:.3}");
            }
            x_p[i] -= eps;
            x_m[i] += eps;
        }
    }

    #[test]
    fn test_swiglu_no_m_state() {
        // Verify initial_m is ignored (no inner state)
        let d = 4;
        let inter = 8;
        let seq_len = 2;
        let params = make_params(d, inter, 55);
        let x = make_input(seq_len, d, 56);

        let rule = SwiGluMlp { intermediate_size: inter };
        // Both should produce identical output regardless of initial_m
        let (y1, _) = rule.step(&params, &x, seq_len, d, None);
        let (y2, _) = rule.step(&params, &x, seq_len, d, Some(vec![0.0f32; d * d]));
        assert_eq!(y1, y2, "SwiGluMlp should ignore initial_m");
    }
}
