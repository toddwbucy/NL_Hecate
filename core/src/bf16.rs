/// bf16 (bfloat16) precision support for mixed-precision storage.
///
/// Projection matrices (W_K, W_V, W_Q) are stored in bf16 for memory savings.
/// The optimizer holds fp32 master copies; after each update, the master is
/// truncated to bf16 for the stored copy. Inner-loop operations cast bf16 → fp32
/// before use, ensuring no precision loss in memory accumulation.
///
/// bf16 format: 1 sign + 8 exponent + 7 mantissa bits (same range as f32, less precision).
/// Conversion rounds the lower 16 mantissa bits to nearest-even before truncation.
///
/// Source: specs/infrastructure/precision/00_numerical_precision.md
/// Constraint: Inner-loop MUST be fp32 (bf16 drift corrupts memory after ~100 steps).

/// Convert f32 to bf16, stored as u16. Rounds lower 16 mantissa bits to nearest-even.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Preserve NaNs explicitly: rounding can collapse some NaN payloads to ±inf.
    if x.is_nan() {
        // Force quiet-NaN: exponent all-ones, mantissa bit set.
        let sign = (bits >> 16) & 0x8000;
        return (sign | 0x7FC0) as u16; // qNaN in bf16
    }
    // Round to nearest even: add 0x7FFF + bit 16 (round-to-even bias)
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

/// Convert bf16 (stored as u16) back to f32. Zero-fills the lower 16 mantissa bits.
#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Bulk convert f32 slice to bf16 vec.
pub fn f32_slice_to_bf16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&x| f32_to_bf16(x)).collect()
}

/// Bulk convert bf16 vec to f32 vec.
pub fn bf16_slice_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&x| bf16_to_f32(x)).collect()
}

/// Mixed-precision storage for projection matrices.
///
/// Holds bf16 stored weights (for memory savings and forward pass dispatch)
/// alongside fp32 master copies (for optimizer updates). After each optimizer
/// step, `sync_from_master()` truncates the master to bf16.
///
/// The inner loop always uses `as_f32()` to get fp32 values — no bf16
/// computation ever enters the memory update path.
pub struct Bf16Storage {
    /// bf16-stored weights (used for forward dispatch, checkpoint serialization).
    stored: Vec<u16>,
    /// fp32 master copy (used by optimizer, source of truth for parameters).
    master: Vec<f32>,
}

impl Bf16Storage {
    /// Create from fp32 weights (initializes both master and bf16 stored copy).
    pub fn from_f32(weights: &[f32]) -> Self {
        Bf16Storage {
            stored: f32_slice_to_bf16(weights),
            master: weights.to_vec(),
        }
    }

    /// Get fp32 view for inner-loop operations (cast from bf16 stored copy).
    /// This is what the memory rules see — always fp32.
    pub fn as_f32(&self) -> Vec<f32> {
        bf16_slice_to_f32(&self.stored)
    }

    /// Get mutable slice of fp32 master copy (for optimizer updates).
    /// Returns a slice (not Vec) to prevent callers from resizing the buffer.
    pub fn master_mut(&mut self) -> &mut [f32] {
        &mut self.master
    }

    /// Get reference to fp32 master copy.
    pub fn master(&self) -> &[f32] {
        &self.master
    }

    /// Sync bf16 stored copy from fp32 master (call after optimizer step).
    pub fn sync_from_master(&mut self) {
        assert_eq!(self.stored.len(), self.master.len(),
            "bf16 stored/master length mismatch: {} vs {}", self.stored.len(), self.master.len());
        for i in 0..self.master.len() {
            self.stored[i] = f32_to_bf16(self.master[i]);
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.stored.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.stored.is_empty()
    }

    /// Memory savings: bf16 stored uses 2 bytes/element vs 4 for fp32.
    /// Total memory: 2*N (bf16 stored) + 4*N (fp32 master) = 6*N bytes.
    /// Without bf16: 4*N (fp32 only). Savings come when master is dropped
    /// at inference (no optimizer), leaving only 2*N bf16 stored.
    pub fn bf16_bytes(&self) -> usize {
        self.stored.len() * 2
    }

    /// Get raw bf16 data (for serialization/checkpoint).
    pub fn stored_bf16(&self) -> &[u16] {
        &self.stored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_round_trip_exact_powers() {
        // Powers of 2 should round-trip exactly (no mantissa bits lost)
        for &v in &[1.0f32, 2.0, 4.0, 0.5, 0.25, -1.0, -8.0] {
            let bf = f32_to_bf16(v);
            let back = bf16_to_f32(bf);
            assert_eq!(back, v, "Exact round-trip failed for {v}");
        }
    }

    #[test]
    fn test_bf16_round_trip_accuracy() {
        // General values: bf16 has 7 mantissa bits → ~1% relative error max
        let values = [0.1, 0.3, 1.5, 3.14, 100.0, -0.7, -42.0, 0.001];
        for &v in &values {
            let bf = f32_to_bf16(v);
            let back = bf16_to_f32(bf);
            let rel_err = ((back - v) / v).abs();
            assert!(rel_err < 0.01,
                "bf16 round-trip error too large for {v}: got {back}, rel_err={rel_err:.4e}");
        }
    }

    #[test]
    fn test_bf16_zero() {
        let bf = f32_to_bf16(0.0);
        assert_eq!(bf16_to_f32(bf), 0.0);
        let bf_neg = f32_to_bf16(-0.0);
        let back_neg = bf16_to_f32(bf_neg);
        assert_eq!(back_neg, 0.0);
        assert!(back_neg.is_sign_negative(), "negative zero sign bit should be preserved");
    }

    #[test]
    fn test_bf16_special_values() {
        // Infinity
        let bf_inf = f32_to_bf16(f32::INFINITY);
        assert!(bf16_to_f32(bf_inf).is_infinite());
        // NaN
        let bf_nan = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(bf_nan).is_nan());
    }

    #[test]
    fn test_bf16_storage_basic() {
        let weights = vec![1.0, 2.0, 3.0, 0.5, -1.0];
        let storage = Bf16Storage::from_f32(&weights);
        assert_eq!(storage.len(), 5);

        // as_f32() should be close to original (these are exact in bf16)
        let f32_view = storage.as_f32();
        assert_eq!(f32_view, weights);
    }

    #[test]
    fn test_bf16_storage_master_update_sync() {
        let weights = vec![1.0, 2.0, 3.0];
        let mut storage = Bf16Storage::from_f32(&weights);

        // Simulate optimizer update on master
        storage.master_mut()[0] = 1.5;
        storage.master_mut()[1] = 2.7;

        // Before sync, bf16 stored still has old values
        let old = storage.as_f32();
        assert_eq!(old[0], 1.0);

        // After sync, bf16 stored reflects updated master
        storage.sync_from_master();
        let new = storage.as_f32();
        let expected_0 = bf16_to_f32(f32_to_bf16(1.5));
        let expected_1 = bf16_to_f32(f32_to_bf16(2.7));
        assert_eq!(new[0], expected_0);
        assert_eq!(new[1], expected_1);
        assert_eq!(new[2], 3.0); // unchanged
    }

    #[test]
    fn test_bf16_storage_inference_savings() {
        // At inference, only bf16 stored needed (2 bytes/element)
        let weights = vec![0.0f32; 1000];
        let storage = Bf16Storage::from_f32(&weights);
        assert_eq!(storage.bf16_bytes(), 2000); // 2 * 1000
    }

    #[test]
    fn test_bf16_bulk_conversion() {
        let src: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let bf16 = f32_slice_to_bf16(&src);
        let back = bf16_slice_to_f32(&bf16);
        assert_eq!(bf16.len(), 100);
        assert_eq!(back.len(), 100);
        for i in 0..100 {
            let rel_err = if src[i].abs() > 1e-6 {
                ((back[i] - src[i]) / src[i]).abs()
            } else {
                (back[i] - src[i]).abs()
            };
            assert!(rel_err < 0.01, "Bulk conversion error at [{i}]: src={}, back={}", src[i], back[i]);
        }
    }

    #[test]
    fn test_bf16_inner_loop_fp32_guarantee() {
        // Verify that casting bf16→f32 produces exact fp32 values
        // (no subnormal or denorm issues for typical weight ranges)
        let weights = vec![0.01, 0.1, 1.0, 10.0, 100.0];
        let storage = Bf16Storage::from_f32(&weights);
        let fp32 = storage.as_f32();
        for &v in &fp32 {
            assert!(v.is_finite(), "Inner-loop fp32 value should be finite");
            // Verify it's a valid f32 (not subnormal for these ranges)
            assert!(v.abs() > f32::MIN_POSITIVE || v == 0.0,
                "Value {v} is subnormal — unexpected for typical weight range");
        }
    }
}
