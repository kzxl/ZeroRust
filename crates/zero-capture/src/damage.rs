//! High-speed SIMD tile damage detector and dirty rectangle tracker.

use crate::rect::Rect;

/// Standard tile dimension in pixels ($32 \times 32$).
pub const TILE_SIZE: u32 = 32;

/// Evaluates if two 32x32 pixel tiles differ using SIMD/fast 64-bit word comparisons.
///
/// # Safety
/// Both `cur_ptr` and `ref_ptr` must point to valid memory buffers with at least `stride * 32` bytes.
#[inline]
pub unsafe fn is_tile_damaged(cur_ptr: *const u8, ref_ptr: *const u8, stride: usize) -> bool {
    let row_bytes = (TILE_SIZE as usize) * 4; // 32 pixels * 4 bytes = 128 bytes per row

    for row in 0..(TILE_SIZE as usize) {
        let c_row = cur_ptr.add(row * stride);
        let r_row = ref_ptr.add(row * stride);

        // Fast 64-bit comparison loop (16 u64 chunks per 128 bytes)
        let c_u64 = c_row as *const u64;
        let r_u64 = r_row as *const u64;

        for chunk in 0..(row_bytes / 8) {
            if *c_u64.add(chunk) != *r_u64.add(chunk) {
                return true;
            }
        }
    }
    false
}

/// Scanner that compares the current frame against a reference frame to locate modified tiles.
pub struct DamageScanner {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
}

impl DamageScanner {
    /// Creates a damage scanner for a fixed resolution.
    pub fn new(width: u32, height: u32) -> Self {
        let tiles_x = width.div_ceil(TILE_SIZE);
        let tiles_y = height.div_ceil(TILE_SIZE);
        Self {
            width,
            height,
            tiles_x,
            tiles_y,
        }
    }

    /// Scans current and reference frame buffers, returning all modified tile rectangles.
    pub fn scan_damage(
        &self,
        current_data: &[u8],
        reference_data: &[u8],
        stride: usize,
    ) -> Vec<Rect> {
        let mut damaged_rects = Vec::new();
        if current_data.len() < stride * (self.height as usize)
            || reference_data.len() < stride * (self.height as usize)
        {
            return damaged_rects;
        }

        for ty in 0..self.tiles_y {
            let py = ty * TILE_SIZE;
            let tile_h = (self.height - py).min(TILE_SIZE);

            for tx in 0..self.tiles_x {
                let px = tx * TILE_SIZE;
                let tile_w = (self.width - px).min(TILE_SIZE);

                let byte_offset = (py as usize) * stride + (px as usize) * 4;

                unsafe {
                    let cur_ptr = current_data.as_ptr().add(byte_offset);
                    let ref_ptr = reference_data.as_ptr().add(byte_offset);

                    let changed = if tile_w == TILE_SIZE && tile_h == TILE_SIZE {
                        is_tile_damaged(cur_ptr, ref_ptr, stride)
                    } else {
                        // Edge boundary tile comparison
                        let mut diff = false;
                        for r in 0..(tile_h as usize) {
                            let c_slice = std::slice::from_raw_parts(
                                cur_ptr.add(r * stride),
                                (tile_w as usize) * 4,
                            );
                            let r_slice = std::slice::from_raw_parts(
                                ref_ptr.add(r * stride),
                                (tile_w as usize) * 4,
                            );
                            if c_slice != r_slice {
                                diff = true;
                                break;
                            }
                        }
                        diff
                    };

                    if changed {
                        damaged_rects.push(Rect {
                            x: px,
                            y: py,
                            width: tile_w,
                            height: tile_h,
                        });
                    }
                }
            }
        }

        damaged_rects
    }

    /// Total number of tiles on X axis.
    pub fn tiles_x(&self) -> u32 {
        self.tiles_x
    }

    /// Total number of tiles on Y axis.
    pub fn tiles_y(&self) -> u32 {
        self.tiles_y
    }

    /// Total tile count for the frame.
    pub fn total_tiles(&self) -> usize {
        (self.tiles_x * self.tiles_y) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_scanner() {
        let width = 64;
        let height = 64;
        let stride = 64 * 4;
        let mut cur = vec![0u8; stride * height];
        let ref_b = vec![0u8; stride * height];

        let scanner = DamageScanner::new(width as u32, height as u32);
        assert_eq!(scanner.total_tiles(), 4); // 2x2 tiles of 32x32

        // Initially zero damage
        let damage = scanner.scan_damage(&cur, &ref_b, stride);
        assert_eq!(damage.len(), 0);

        // Mutate top-left tile
        cur[0] = 255;
        let damage = scanner.scan_damage(&cur, &ref_b, stride);
        assert_eq!(damage.len(), 1);
        assert_eq!(damage[0], Rect::new(0, 0, 32, 32));
    }
}
