//! Cacheline-aligned pixel buffers and lock-free frame buffer pool.

use std::sync::atomic::{AtomicBool, Ordering};

/// CPU Cacheline alignment in bytes for high-speed SIMD loads.
pub const CACHELINE_ALIGN_BYTES: usize = 64;

/// Standard OS memory page size (4096 bytes).
pub const PAGE_SIZE_BYTES: usize = 4096;

/// Uncompressed pixel arrangement format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 32-bit Blue-Green-Red-Alpha (standard native Windows D3D / GDI format).
    Bgra8888,
    /// 32-bit Red-Green-Blue-Alpha (standard OpenGL / WebGL / Wayland format).
    Rgba8888,
    /// 24-bit Red-Green-Blue packed.
    Rgb888,
    /// Planar YUV 4:2:0 subsampled format for high-compression video modes.
    Yuv420p,
}

impl PixelFormat {
    /// Returns bytes per pixel for packed formats, or 0 for planar formats.
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Bgra8888 | Self::Rgba8888 => 4,
            Self::Rgb888 => 3,
            Self::Yuv420p => 0,
        }
    }
}

/// Aligned contiguous pixel frame buffer.
pub struct FrameBuffer {
    /// Unique pool slot identifier.
    pub id: usize,
    /// Width of the frame in pixels.
    pub width: u32,
    /// Height of the frame in pixels.
    pub height: u32,
    /// Stride in bytes per scanline (cacheline-aligned).
    pub stride: usize,
    /// Active pixel format.
    pub format: PixelFormat,
    /// Total allocated byte capacity.
    capacity: usize,
    /// Aligned raw byte pointer.
    data_ptr: *mut u8,
    /// In-use atomic flag for lock-free pool slot acquisition.
    in_use: AtomicBool,
}

// Safety: FrameBuffer owns its memory and is guarded by in_use AtomicBool.
unsafe impl Send for FrameBuffer {}
unsafe impl Sync for FrameBuffer {}

impl FrameBuffer {
    /// Returns raw immutable pixel slice.
    pub fn as_slice(&self) -> &[u8] {
        let size = self.stride * (self.height as usize);
        unsafe { std::slice::from_raw_parts(self.data_ptr, size.min(self.capacity)) }
    }

    /// Returns raw pointer to memory buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.data_ptr
    }

    /// Returns raw mutable pointer to memory buffer.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.data_ptr
    }

    /// Total byte length of active frame data.
    pub fn len(&self) -> usize {
        self.stride * (self.height as usize)
    }

    /// Checks if frame buffer has zero dimensions.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

impl Drop for FrameBuffer {
    fn drop(&mut self) {
        if !self.data_ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.capacity, PAGE_SIZE_BYTES)
                .expect("Valid alignment");
            unsafe {
                std::alloc::dealloc(self.data_ptr, layout);
            }
        }
    }
}

/// RAII Guard that releases the frame buffer back to the pool upon drop.
pub struct FrameGuard<'a> {
    buffer: &'a FrameBuffer,
}

impl<'a> FrameGuard<'a> {
    /// Returns mutable slice to the frame's pixel data.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let size = self.buffer.stride * (self.buffer.height as usize);
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.data_ptr, size.min(self.buffer.capacity))
        }
    }

    /// Returns immutable slice to the frame's pixel data.
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

impl<'a> std::ops::Deref for FrameGuard<'a> {
    type Target = FrameBuffer;
    fn deref(&self) -> &Self::Target {
        self.buffer
    }
}

impl<'a> Drop for FrameGuard<'a> {
    fn drop(&mut self) {
        self.buffer.in_use.store(false, Ordering::Release);
    }
}

/// Pre-allocated lock-free pool of frame buffers.
pub struct FrameBufferPool {
    pool: Vec<FrameBuffer>,
    max_width: u32,
    max_height: u32,
}

impl FrameBufferPool {
    /// Creates a new pool with `count` pre-allocated buffers sized for `max_width` x `max_height`.
    pub fn new(count: usize, max_width: u32, max_height: u32, format: PixelFormat) -> Self {
        let bpp = format.bytes_per_pixel().max(4);
        let raw_stride = (max_width as usize) * bpp;
        // Align stride to 64-byte cacheline
        let stride = (raw_stride + (CACHELINE_ALIGN_BYTES - 1)) & !(CACHELINE_ALIGN_BYTES - 1);
        let buffer_capacity = stride * (max_height as usize);

        let mut pool = Vec::with_capacity(count);
        for id in 0..count {
            let layout = std::alloc::Layout::from_size_align(buffer_capacity, PAGE_SIZE_BYTES)
                .expect("Valid alignment");
            let data_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!data_ptr.is_null(), "Frame buffer memory allocation failed");

            pool.push(FrameBuffer {
                id,
                width: max_width,
                height: max_height,
                stride,
                format,
                capacity: buffer_capacity,
                data_ptr,
                in_use: AtomicBool::new(false),
            });
        }

        Self {
            pool,
            max_width,
            max_height,
        }
    }

    /// Acquires an idle buffer from the pool without any runtime heap allocation.
    pub fn acquire(&self) -> Option<FrameGuard<'_>> {
        for fb in &self.pool {
            if !fb.in_use.swap(true, Ordering::AcqRel) {
                return Some(FrameGuard { buffer: fb });
            }
        }
        None // Pool exhausted (caller can drop frame or wait)
    }

    /// Returns pool slot capacity count.
    pub fn count(&self) -> usize {
        self.pool.len()
    }

    /// Maximum supported frame width.
    pub fn max_width(&self) -> u32 {
        self.max_width
    }

    /// Maximum supported frame height.
    pub fn max_height(&self) -> u32 {
        self.max_height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_buffer_pool_lifecycle() {
        let pool = FrameBufferPool::new(2, 640, 480, PixelFormat::Bgra8888);
        assert_eq!(pool.count(), 2);

        let g1 = pool.acquire();
        assert!(g1.is_some());
        let g2 = pool.acquire();
        assert!(g2.is_some());

        // Pool is now full (2 acquired)
        let g3 = pool.acquire();
        assert!(g3.is_none());

        // Drop one buffer
        drop(g1);

        // Can acquire again
        let g4 = pool.acquire();
        assert!(g4.is_some());
    }
}
