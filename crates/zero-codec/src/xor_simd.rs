//! High-performance 64-bit word XOR delta difference calculation.

/// Result of XOR delta analysis between current and reference tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XorAnalysis {
    /// True if the current tile is 100% identical to the reference tile.
    pub is_identical: bool,
    /// True if all 1024 pixels in the tile share an identical solid 32-bit color.
    pub is_solid: bool,
    /// Solid 32-bit pixel value `[B, G, R, A]` if `is_solid` is true.
    pub solid_color: u32,
    /// Count of mutated 64-bit words in this tile.
    pub mutated_words: u32,
}

/// Computes the XOR delta between current and reference 32x32 tiles.
///
/// # Safety
/// `cur_ptr` and `ref_ptr` must point to valid image memory of at least `stride * 32` bytes.
/// `out_delta` must have at least 4096 bytes available.
pub unsafe fn compute_tile_xor(
    cur_ptr: *const u8,
    ref_ptr: *const u8,
    out_delta: *mut u8,
    stride: usize,
) -> XorAnalysis {
    let mut mutated_words = 0u32;
    let mut is_solid = true;
    let first_pixel = *(cur_ptr as *const u32);
    let solid_u64 = ((first_pixel as u64) << 32) | (first_pixel as u64);

    for row in 0..32 {
        let c_u64 = cur_ptr.add(row * stride) as *const u64;
        let r_u64 = ref_ptr.add(row * stride) as *const u64;
        let o_u64 = out_delta.add(row * 128) as *mut u64;

        // 16 x 64-bit words per row (128 bytes) = 4 iterations of 4-wide unrolling
        for w in (0..16).step_by(4) {
            let c0 = *c_u64.add(w);
            let r0 = *r_u64.add(w);
            let x0 = c0 ^ r0;
            *o_u64.add(w) = x0;

            let c1 = *c_u64.add(w + 1);
            let r1 = *r_u64.add(w + 1);
            let x1 = c1 ^ r1;
            *o_u64.add(w + 1) = x1;

            let c2 = *c_u64.add(w + 2);
            let r2 = *r_u64.add(w + 2);
            let x2 = c2 ^ r2;
            *o_u64.add(w + 2) = x2;

            let c3 = *c_u64.add(w + 3);
            let r3 = *r_u64.add(w + 3);
            let x3 = c3 ^ r3;
            *o_u64.add(w + 3) = x3;

            // Branchless mutation accumulator (compiled to setne / cmov)
            mutated_words +=
                (x0 != 0) as u32 + (x1 != 0) as u32 + (x2 != 0) as u32 + (x3 != 0) as u32;

            // In-flight dual-pixel solid color comparison eliminating secondary read pass
            if is_solid
                && ((c0 ^ solid_u64) | (c1 ^ solid_u64) | (c2 ^ solid_u64) | (c3 ^ solid_u64)) != 0
            {
                is_solid = false;
            }
        }
    }

    XorAnalysis {
        is_identical: mutated_words == 0,
        is_solid,
        solid_color: first_pixel,
        mutated_words,
    }
}

/// Applies the XOR delta to a reference tile to reconstruct the current tile.
///
/// # Safety
/// `ref_ptr` and `out_ptr` must point to valid image memory of at least `stride * 32` bytes.
/// `delta_ptr` must point to at least 4096 bytes of valid delta data.
pub unsafe fn apply_tile_xor(
    ref_ptr: *const u8,
    delta_ptr: *const u8,
    out_ptr: *mut u8,
    stride: usize,
) {
    for row in 0..32 {
        let r_u64 = ref_ptr.add(row * stride) as *const u64;
        let d_u64 = delta_ptr.add(row * 128) as *const u64;
        let o_u64 = out_ptr.add(row * stride) as *mut u64;

        // 4-wide unrolling enables 256-bit AVX2 / 128-bit SSE auto-vectorization
        for w in (0..16).step_by(4) {
            *o_u64.add(w) = *r_u64.add(w) ^ *d_u64.add(w);
            *o_u64.add(w + 1) = *r_u64.add(w + 1) ^ *d_u64.add(w + 1);
            *o_u64.add(w + 2) = *r_u64.add(w + 2) ^ *d_u64.add(w + 2);
            *o_u64.add(w + 3) = *r_u64.add(w + 3) ^ *d_u64.add(w + 3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::TILE_RAW_BYTES;

    #[test]
    fn test_compute_and_apply_tile_xor() {
        let stride = 128;
        let mut cur = vec![0u8; stride * 32];
        let ref_b = vec![0u8; stride * 32];
        let mut delta = vec![0u8; TILE_RAW_BYTES];
        let mut reconstructed = vec![0u8; stride * 32];

        // Fill current with a pattern
        for (i, b) in cur.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }

        unsafe {
            let analysis =
                compute_tile_xor(cur.as_ptr(), ref_b.as_ptr(), delta.as_mut_ptr(), stride);
            assert!(!analysis.is_identical);
            assert_eq!(analysis.mutated_words, 16 * 32);

            apply_tile_xor(
                ref_b.as_ptr(),
                delta.as_ptr(),
                reconstructed.as_mut_ptr(),
                stride,
            );
        }

        assert_eq!(cur, reconstructed);
    }
}
