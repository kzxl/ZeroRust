//! Connected Component Labeling (CCL) and Blob Analysis.

use crate::image::ImageBuffer;

/// Bounding rectangle of a detected blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BoundingBox {
    /// Minimum X coordinate.
    pub min_x: usize,
    /// Minimum Y coordinate.
    pub min_y: usize,
    /// Maximum X coordinate.
    pub max_x: usize,
    /// Maximum Y coordinate.
    pub max_y: usize,
}

impl BoundingBox {
    /// Returns the box width in pixels.
    #[inline]
    pub const fn width(&self) -> usize {
        self.max_x.saturating_sub(self.min_x) + 1
    }

    /// Returns the box height in pixels.
    #[inline]
    pub const fn height(&self) -> usize {
        self.max_y.saturating_sub(self.min_y) + 1
    }
}

/// A segmented connected component (Blob) with spatial metrics.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Blob {
    /// Component unique label ID.
    pub label: u16,
    /// Total pixel count (area).
    pub area: usize,
    /// Bounding rectangle.
    pub bbox: BoundingBox,
    /// Horizontal centroid center of mass.
    pub centroid_x: f64,
    /// Vertical centroid center of mass.
    pub centroid_y: f64,
}

/// Connected Component Analyzer without dynamic heap allocation.
pub struct BlobAnalyzer;

impl BlobAnalyzer {
    /// Extracts connected components from a binary image (foreground = 255).
    ///
    /// * `binary_image`: Input binary image.
    /// * `labels_scratch`: Temporary scratch buffer for label mapping (must have length >= `binary_image.len()`).
    /// * `out_blobs`: Fixed-size output array for detected blobs.
    ///
    /// Returns the actual number of detected blobs.
    pub fn analyze<const MAX_PIXELS: usize, const MAX_BLOBS: usize>(
        binary_image: &ImageBuffer<MAX_PIXELS>,
        labels_scratch: &mut [u16],
        out_blobs: &mut [Blob; MAX_BLOBS],
    ) -> usize {
        let w = binary_image.width();
        let h = binary_image.height();
        let num_pixels = w * h;

        if labels_scratch.len() < num_pixels || num_pixels == 0 {
            return 0;
        }

        labels_scratch[..num_pixels].fill(0);

        // Union-find parent table
        let mut parent = [0u16; 64]; // Max 64 concurrent root components
        for (i, p) in parent.iter_mut().enumerate() {
            *p = i as u16;
        }

        let find_root = |mut label: u16, parent: &[u16]| -> u16 {
            while (label as usize) < parent.len() && parent[label as usize] != label {
                label = parent[label as usize];
            }
            label
        };

        let union_roots = |a: u16, b: u16, parent: &mut [u16]| {
            let root_a = find_root(a, parent);
            let root_b = find_root(b, parent);
            if root_a != root_b && (root_b as usize) < parent.len() {
                parent[root_b as usize] = root_a;
            }
        };

        let mut next_label = 1u16;

        // Pass 1: Initial provisional labeling
        for y in 0..h {
            for x in 0..w {
                if binary_image.get_pixel(x, y) == 0 {
                    continue;
                }

                let west = if x > 0 {
                    labels_scratch[y * w + (x - 1)]
                } else {
                    0
                };
                let north = if y > 0 {
                    labels_scratch[(y - 1) * w + x]
                } else {
                    0
                };

                match (west > 0, north > 0) {
                    (false, false) => {
                        if (next_label as usize) < parent.len() {
                            labels_scratch[y * w + x] = next_label;
                            next_label += 1;
                        }
                    }
                    (true, false) => labels_scratch[y * w + x] = west,
                    (false, true) => labels_scratch[y * w + x] = north,
                    (true, true) => {
                        labels_scratch[y * w + x] = west;
                        if west != north {
                            union_roots(west, north, &mut parent);
                        }
                    }
                }
            }
        }

        // Pass 2: Canonical root resolution and spatial moments accumulation
        let mut blob_count = 0;
        let mut label_to_idx = [0usize; 64];

        for item in out_blobs.iter_mut() {
            *item = Blob::default();
        }

        for y in 0..h {
            for x in 0..w {
                let lbl = labels_scratch[y * w + x];
                if lbl == 0 {
                    continue;
                }

                let root = find_root(lbl, &parent);
                labels_scratch[y * w + x] = root;

                let idx = if label_to_idx[root as usize] == 0 {
                    if blob_count >= MAX_BLOBS {
                        continue;
                    }
                    blob_count += 1;
                    label_to_idx[root as usize] = blob_count;
                    let b = &mut out_blobs[blob_count - 1];
                    b.label = root;
                    b.bbox.min_x = x;
                    b.bbox.max_x = x;
                    b.bbox.min_y = y;
                    b.bbox.max_y = y;
                    blob_count - 1
                } else {
                    label_to_idx[root as usize] - 1
                };

                let b = &mut out_blobs[idx];
                b.area += 1;
                b.centroid_x += x as f64;
                b.centroid_y += y as f64;
                b.bbox.min_x = b.bbox.min_x.min(x);
                b.bbox.max_x = b.bbox.max_x.max(x);
                b.bbox.min_y = b.bbox.min_y.min(y);
                b.bbox.max_y = b.bbox.max_y.max(y);
            }
        }

        // Compute final centroids
        for b in out_blobs.iter_mut().take(blob_count) {
            if b.area > 0 {
                b.centroid_x /= b.area as f64;
                b.centroid_y /= b.area as f64;
            }
        }

        blob_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_two_islands() {
        let mut img: ImageBuffer<64> = ImageBuffer::new(8, 8);

        // Island 1 at top-left (2x2 square)
        img.set_pixel(1, 1, 255);
        img.set_pixel(2, 1, 255);
        img.set_pixel(1, 2, 255);
        img.set_pixel(2, 2, 255);

        // Island 2 at bottom-right (single pixel)
        img.set_pixel(6, 6, 255);

        let mut scratch = [0u16; 64];
        let mut blobs = [Blob::default(); 8];
        let count = BlobAnalyzer::analyze(&img, &mut scratch, &mut blobs);

        assert_eq!(count, 2);
        // Find island 1
        let b1 = blobs.iter().find(|b| b.area == 4).unwrap();
        assert_eq!(b1.bbox.min_x, 1);
        assert_eq!(b1.bbox.max_x, 2);
        assert_eq!(b1.bbox.min_y, 1);
        assert_eq!(b1.bbox.max_y, 2);
        assert_eq!(b1.centroid_x, 1.5);
        assert_eq!(b1.centroid_y, 1.5);

        // Find island 2
        let b2 = blobs.iter().find(|b| b.area == 1).unwrap();
        assert_eq!(b2.bbox.min_x, 6);
        assert_eq!(b2.bbox.max_x, 6);
    }
}
