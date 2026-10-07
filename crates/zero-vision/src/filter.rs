//! Spatial convolution filters: Sobel edge detection and Box blur.

use crate::image::ImageBuffer;

/// Applies a 3x3 Sobel gradient operator to extract edges.
pub fn sobel_filter<const MAX_PIXELS: usize>(
    src: &ImageBuffer<MAX_PIXELS>,
    dst: &mut ImageBuffer<MAX_PIXELS>,
) {
    let w = src.width();
    let h = src.height();
    *dst = ImageBuffer::new(w, h);

    if w < 3 || h < 3 {
        return;
    }

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let p00 = src.get_pixel(x - 1, y - 1) as i32;
            let p01 = src.get_pixel(x, y - 1) as i32;
            let p02 = src.get_pixel(x + 1, y - 1) as i32;

            let p10 = src.get_pixel(x - 1, y) as i32;
            let p12 = src.get_pixel(x + 1, y) as i32;

            let p20 = src.get_pixel(x - 1, y + 1) as i32;
            let p21 = src.get_pixel(x, y + 1) as i32;
            let p22 = src.get_pixel(x + 1, y + 1) as i32;

            // Gx = [-1 0 1; -2 0 2; -1 0 1]
            let gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);

            // Gy = [-1 -2 -1; 0 0 0; 1 2 1]
            let gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);

            let magnitude = (gx.abs() + gy.abs()).min(255) as u8;
            dst.set_pixel(x, y, magnitude);
        }
    }
}

/// Applies a 3x3 spatial Box blur filter for image denoising.
pub fn box_blur<const MAX_PIXELS: usize>(
    src: &ImageBuffer<MAX_PIXELS>,
    dst: &mut ImageBuffer<MAX_PIXELS>,
) {
    let w = src.width();
    let h = src.height();
    *dst = ImageBuffer::new(w, h);

    if w < 3 || h < 3 {
        dst.copy_from_raw(src.as_slice()).ok();
        return;
    }

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let mut sum: u32 = 0;
            for dy in 0..3 {
                for dx in 0..3 {
                    sum += src.get_pixel(x - 1 + dx, y - 1 + dy) as u32;
                }
            }
            dst.set_pixel(x, y, (sum / 9) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobel_vertical_edge() {
        let mut img: ImageBuffer<36> = ImageBuffer::new(6, 6);
        // Sharp vertical edge down the middle: left black (0), right white (255)
        for y in 0..6 {
            for x in 3..6 {
                img.set_pixel(x, y, 255);
            }
        }

        let mut edge = ImageBuffer::new(6, 6);
        sobel_filter(&img, &mut edge);

        // Pixels right on the boundary should have high gradient response
        assert!(edge.get_pixel(2, 2) > 100);
        assert!(edge.get_pixel(3, 2) > 100);
        // Pixels deep inside uniform white or black should have zero response
        assert_eq!(edge.get_pixel(0, 0), 0);
    }
}
