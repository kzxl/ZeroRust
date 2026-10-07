//! Const-generic, zero-heap grayscale image buffer for embedded vision.

use zero_core::error::{ZeroError, ZeroResult};

/// Fixed-capacity grayscale (Mono8) image buffer without dynamic heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageBuffer<const MAX_PIXELS: usize> {
    width: usize,
    height: usize,
    pixels: [u8; MAX_PIXELS],
}

impl<const MAX_PIXELS: usize> Default for ImageBuffer<MAX_PIXELS> {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixels: [0u8; MAX_PIXELS],
        }
    }
}

impl<const MAX_PIXELS: usize> ImageBuffer<MAX_PIXELS> {
    /// Creates a new image buffer with specified width and height.
    pub fn new(width: usize, height: usize) -> Self {
        assert!(
            width * height <= MAX_PIXELS,
            "Dimensions exceed buffer capacity"
        );
        Self {
            width,
            height,
            pixels: [0u8; MAX_PIXELS],
        }
    }

    /// Returns the image width in pixels.
    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height in pixels.
    #[inline]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns total active pixel count.
    #[inline]
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Returns true if image has zero dimensions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the pixel value at `(x, y)`.
    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x]
        } else {
            0
        }
    }

    /// Sets the pixel value at `(x, y)`.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, val: u8) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = val;
        }
    }

    /// Fills active image region with a constant pixel value.
    pub fn fill(&mut self, val: u8) {
        let active_len = self.width * self.height;
        self.pixels[..active_len].fill(val);
    }

    /// Returns an immutable slice of the active raw pixel bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.pixels[..self.width * self.height]
    }

    /// Returns a mutable slice of the active raw pixel bytes.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let active_len = self.width * self.height;
        &mut self.pixels[..active_len]
    }

    /// Copies pixels from a raw byte slice.
    pub fn copy_from_raw(&mut self, raw: &[u8]) -> ZeroResult<()> {
        let active_len = self.width * self.height;
        if raw.len() != active_len {
            return Err(ZeroError::InvalidArgument);
        }
        self.pixels[..active_len].copy_from_slice(raw);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_buffer_get_set() {
        let mut img: ImageBuffer<64> = ImageBuffer::new(8, 8);
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 8);
        assert_eq!(img.get_pixel(3, 4), 0);

        img.set_pixel(3, 4, 255);
        assert_eq!(img.get_pixel(3, 4), 255);
    }
}
