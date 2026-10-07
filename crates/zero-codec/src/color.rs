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

    let (bgra_chunks, _) = bgra[..total_pixels * 4].as_chunks::<4>();
    for (chunk, (y_out, (u_out, v_out))) in bgra_chunks.iter().zip(
        y_plane[..total_pixels].iter_mut().zip(
            u_plane[..total_pixels]
                .iter_mut()
                .zip(v_plane[..total_pixels].iter_mut()),
        ),
    ) {
        let b = chunk[0] as i32;
        let g = chunk[1] as i32;
        let r = chunk[2] as i32;

        // ITU-R BT.601 integer fixed-point matrix:
        // Y = (66*R + 129*G + 25*B + 128) >> 8 + 16
        // U = (-38*R - 74*G + 112*B + 128) >> 8 + 128
        // V = (112*R - 94*G - 18*B + 128) >> 8 + 128
        let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

        *y_out = y.clamp(0, 255) as u8;
        *u_out = u.clamp(0, 255) as u8;
        *v_out = v.clamp(0, 255) as u8;
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

    let (bgra_chunks_mut, _) = bgra[..total_pixels * 4].as_chunks_mut::<4>();
    for (chunk, (&y, (&u, &v))) in bgra_chunks_mut.iter_mut().zip(
        y_plane[..total_pixels].iter().zip(
            u_plane[..total_pixels]
                .iter()
                .zip(v_plane[..total_pixels].iter()),
        ),
    ) {
        let c = y as i32 - 16;
        let d = u as i32 - 128;
        let e = v as i32 - 128;

        let r = (298 * c + 409 * e + 128) >> 8;
        let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        let b = (298 * c + 516 * d + 128) >> 8;

        chunk[0] = b.clamp(0, 255) as u8;
        chunk[1] = g.clamp(0, 255) as u8;
        chunk[2] = r.clamp(0, 255) as u8;
        chunk[3] = 0xFF; // Full opacity
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
