//! Headless mock capturer for testing without display hardware.

use crate::buffer::FrameBufferPool;
use crate::capturer::{CaptureError, CapturedFrame, ScreenCapturer};
use crate::damage::DamageScanner;
use crate::rect::{DisplayInfo, Rect};

/// Deterministic mock capturer that produces simulated moving desktop patterns.
pub struct MockCapturer {
    info: DisplayInfo,
    frame_count: u64,
    prev_buffer: Vec<u8>,
    scanner: DamageScanner,
}

impl MockCapturer {
    /// Creates a mock capturer with custom resolution.
    pub fn new(width: u32, height: u32) -> Self {
        let stride = (width as usize) * 4;
        let prev_buffer = vec![0u8; stride * (height as usize)];
        let scanner = DamageScanner::new(width, height);
        Self {
            info: DisplayInfo {
                id: 0,
                name: "Mock Virtual Display".to_string(),
                width,
                height,
                offset_x: 0,
                offset_y: 0,
                refresh_rate_hz: 60,
                is_primary: true,
                scale_factor_pct: 100,
            },
            frame_count: 0,
            prev_buffer,
            scanner,
        }
    }
}

impl ScreenCapturer for MockCapturer {
    fn capture_frame<'a>(
        &mut self,
        pool: &'a FrameBufferPool,
        _timeout_ms: u32,
    ) -> Result<CapturedFrame<'a>, CaptureError> {
        let mut guard = pool.acquire().ok_or(CaptureError::BufferPoolExhausted)?;
        let stride = guard.stride;
        let w = self.info.width as usize;
        let h = self.info.height as usize;
        let slice = guard.as_mut_slice();

        // Procedural moving block animation to simulate desktop motion
        let block_x = ((self.frame_count * 8) % (w as u64 - 64).max(1)) as usize;
        let block_y = ((self.frame_count * 4) % (h as u64 - 64).max(1)) as usize;

        // Fill background with dark navy (0x10, 0x14, 0x20, 0xFF)
        for y in 0..h {
            let row_offset = y * stride;
            for x in 0..w {
                let p = row_offset + x * 4;
                if x >= block_x && x < block_x + 64 && y >= block_y && y < block_y + 64 {
                    // Moving cyan block (BGRA: 212, 182, 6, 255)
                    slice[p] = 212;
                    slice[p + 1] = 182;
                    slice[p + 2] = 6;
                    slice[p + 3] = 255;
                } else {
                    slice[p] = 0x20;
                    slice[p + 1] = 0x14;
                    slice[p + 2] = 0x10;
                    slice[p + 3] = 0xFF;
                }
            }
        }

        let is_keyframe = self.frame_count == 0;
        let damage_rects = if is_keyframe {
            vec![Rect::new(0, 0, self.info.width, self.info.height)]
        } else {
            self.scanner.scan_damage(slice, &self.prev_buffer, stride)
        };

        // Cache previous frame
        let copy_len = self.prev_buffer.len().min(slice.len());
        self.prev_buffer[..copy_len].copy_from_slice(&slice[..copy_len]);
        self.frame_count += 1;

        Ok(CapturedFrame {
            buffer: guard,
            damage_rects,
            is_keyframe,
            timestamp_ns: self.frame_count * 16_666_666, // ~60 FPS
        })
    }

    fn display_info(&self) -> DisplayInfo {
        self.info.clone()
    }

    fn display_id(&self) -> usize {
        self.info.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::PixelFormat;

    #[test]
    fn test_mock_capturer_loop() {
        let mut capturer = MockCapturer::new(320, 240);
        let pool = FrameBufferPool::new(2, 320, 240, PixelFormat::Bgra8888);

        // Frame 0: Keyframe
        let f1 = capturer.capture_frame(&pool, 100).expect("Capture f1");
        assert!(f1.is_keyframe);
        assert_eq!(f1.damage_rects.len(), 1);
        drop(f1);

        // Frame 1: Delta frame
        let f2 = capturer.capture_frame(&pool, 100).expect("Capture f2");
        assert!(!f2.is_keyframe);
        assert!(!f2.damage_rects.is_empty());
    }
}
