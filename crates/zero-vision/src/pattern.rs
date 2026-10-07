//! Industrial fiducial marker and crosshair pattern locator.

use crate::image::ImageBuffer;

/// 2D Point coordinate in image space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ImagePoint {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

/// Fiducial crosshair and alignment pattern detector.
pub struct CrosshairDetector;

impl CrosshairDetector {
    /// Detects a crosshair pattern by computing row and column intensity projection profiles.
    ///
    /// Useful for automated PCB alignment and robotic visual servoing.
    pub fn locate<const MAX_PIXELS: usize>(
        image: &ImageBuffer<MAX_PIXELS>,
        invert: bool,
    ) -> Option<ImagePoint> {
        let w = image.width();
        let h = image.height();
        if w == 0 || h == 0 || w > 1024 || h > 1024 {
            return None;
        }

        let mut row_sums = [0u32; 1024];
        let mut col_sums = [0u32; 1024];

        for (y, r) in row_sums.iter_mut().take(h).enumerate() {
            for (x, c) in col_sums.iter_mut().take(w).enumerate() {
                let p = image.get_pixel(x, y) as u32;
                *r += p;
                *c += p;
            }
        }

        let best_row = if invert {
            row_sums[..h]
                .iter()
                .enumerate()
                .min_by_key(|&(_, &val)| val)?
                .0
        } else {
            row_sums[..h]
                .iter()
                .enumerate()
                .max_by_key(|&(_, &val)| val)?
                .0
        };

        let best_col = if invert {
            col_sums[..w]
                .iter()
                .enumerate()
                .min_by_key(|&(_, &val)| val)?
                .0
        } else {
            col_sums[..w]
                .iter()
                .enumerate()
                .max_by_key(|&(_, &val)| val)?
                .0
        };

        Some(ImagePoint {
            x: best_col as f64,
            y: best_row as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosshair_center_detection() {
        let mut img: ImageBuffer<81> = ImageBuffer::new(9, 9);

        // Draw a bright crosshair centered at (4, 4)
        for i in 0..9 {
            img.set_pixel(i, 4, 255); // Horizontal line
            img.set_pixel(4, i, 255); // Vertical line
        }

        let center = CrosshairDetector::locate(&img, false).unwrap();
        assert_eq!(center.x, 4.0);
        assert_eq!(center.y, 4.0);
    }
}
