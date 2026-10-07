//! Fast fixed-point integer color space conversion routines.

/// Converts BGRA 32bpp packed pixels into planar YUV444p.
pub fn bgra_to_yuv444p(
    bgra: &[u8],
    width: usize,
    height: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
) {
    let total_pixels = width * height;
    assert!(bgra.len() >= total_pixels * 4);
    assert!(y_plane.len() >= total_pixels);
    assert!(u_plane.len() >= total_pixels);
    assert!(v_plane.len() >= total_pixels);

    for i in 0..total_pixels {
        let b = bgra[i * 4] as i32;
        let g = bgra[i * 4 + 1] as i32;
        let r = bgra[i * 4 + 2] as i32;

        // ITU-R BT.601 integer fixed-point matrix:
        // Y = (66*R + 129*G + 25*B + 128) >> 8 + 16
        // U = (-38*R - 74*G + 112*B + 128) >> 8 + 128
        // V = (112*R - 94*G - 18*B + 128) >> 8 + 128
        let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

        y_plane[i] = y.clamp(0, 255) as u8;
        u_plane[i] = u.clamp(0, 255) as u8;
        v_plane[i] = v.clamp(0, 255) as u8;
    }
}

/// Converts planar YUV444p back into BGRA 32bpp packed pixels.
pub fn yuv444p_to_bgra(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
    bgra: &mut [u8],
) {
    let total_pixels = width * height;
    assert!(bgra.len() >= total_pixels * 4);
    assert!(y_plane.len() >= total_pixels);
    assert!(u_plane.len() >= total_pixels);
    assert!(v_plane.len() >= total_pixels);

    for i in 0..total_pixels {
        let c = y_plane[i] as i32 - 16;
        let d = u_plane[i] as i32 - 128;
        let e = v_plane[i] as i32 - 128;

        let r = (298 * c + 409 * e + 128) >> 8;
        let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        let b = (298 * c + 516 * d + 128) >> 8;

        bgra[i * 4] = b.clamp(0, 255) as u8;
        bgra[i * 4 + 1] = g.clamp(0, 255) as u8;
        bgra[i * 4 + 2] = r.clamp(0, 255) as u8;
        bgra[i * 4 + 3] = 0xFF; // Full opacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgra_yuv444p_roundtrip() {
        let w = 16;
        let h = 16;
        let mut original_bgra = vec![0u8; w * h * 4];
        for i in 0..w * h {
            original_bgra[i * 4] = 64; // B
            original_bgra[i * 4 + 1] = 128; // G
            original_bgra[i * 4 + 2] = 192; // R
            original_bgra[i * 4 + 3] = 255; // A
        }

        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; w * h];
        let mut v = vec![0u8; w * h];
        bgra_to_yuv444p(&original_bgra, w, h, &mut y, &mut u, &mut v);

        let mut reconstructed = vec![0u8; w * h * 4];
        yuv444p_to_bgra(&y, &u, &v, w, h, &mut reconstructed);

        // Allow ±3 rounding delta due to fixed point conversion
        for i in 0..w * h * 4 {
            let diff = (original_bgra[i] as i32 - reconstructed[i] as i32).abs();
            assert!(
                diff <= 3,
                "Channel {} mismatch: orig={}, recon={}",
                i % 4,
                original_bgra[i],
                reconstructed[i]
            );
        }
    }
}
