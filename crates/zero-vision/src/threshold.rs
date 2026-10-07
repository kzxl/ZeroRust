//! Otsu's automatic optimal thresholding and binary segmentation.

use crate::image::ImageBuffer;

/// Computes the optimal global binarization threshold using Otsu's method.
pub fn otsu_threshold<const MAX_PIXELS: usize>(image: &ImageBuffer<MAX_PIXELS>) -> u8 {
    let mut histogram = [0u32; 256];
    for &p in image.as_slice() {
        histogram[p as usize] += 1;
    }

    let total = image.len() as f64;
    if total == 0.0 {
        return 128;
    }

    let mut sum_all: f64 = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_all += (i as f64) * (count as f64);
    }

    let mut weight_bg: f64 = 0.0;
    let mut sum_bg: f64 = 0.0;
    let mut max_variance: f64 = 0.0;
    let mut optimal_threshold: u8 = 128;

    for (t, &count) in histogram.iter().enumerate().take(256) {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }

        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += (t as f64) * (count as f64);

        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_all - sum_bg) / weight_fg;

        let diff = mean_bg - mean_fg;
        let between_class_variance = weight_bg * weight_fg * diff * diff;

        if between_class_variance > max_variance {
            max_variance = between_class_variance;
            optimal_threshold = t as u8;
        }
    }

    optimal_threshold
}

/// Applies binary thresholding from `src` into `dst`.
///
/// If `invert` is true, pixels above threshold become 0 and pixels below become 255.
pub fn binarize<const MAX_PIXELS: usize>(
    src: &ImageBuffer<MAX_PIXELS>,
    dst: &mut ImageBuffer<MAX_PIXELS>,
    threshold: u8,
    invert: bool,
) {
    let w = src.width();
    let h = src.height();
    *dst = ImageBuffer::new(w, h);

    let (high, low) = if invert { (0, 255) } else { (255, 0) };

    for y in 0..h {
        for x in 0..w {
            let p = src.get_pixel(x, y);
            let out = if p > threshold { high } else { low };
            dst.set_pixel(x, y, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otsu_bimodal_image() {
        let mut img: ImageBuffer<100> = ImageBuffer::new(10, 10);
        // Half black (30), half white (220)
        for y in 0..10 {
            for x in 0..10 {
                if x < 5 {
                    img.set_pixel(x, y, 30);
                } else {
                    img.set_pixel(x, y, 220);
                }
            }
        }

        let th = otsu_threshold(&img);
        // Otsu threshold should cleanly sit between 30 and 220
        assert!((30..=220).contains(&th));

        let mut binarized = ImageBuffer::new(10, 10);
        binarize(&img, &mut binarized, th, false);
        assert_eq!(binarized.get_pixel(0, 0), 0);
        assert_eq!(binarized.get_pixel(9, 9), 255);
    }
}
