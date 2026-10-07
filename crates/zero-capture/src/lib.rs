//! # ZeroCapture
//!
//! Ultra-low-latency, zero-copy screen capture and frame buffer pool for ZConn in ZeroRust:
//! - Pre-allocated, cacheline-aligned (64-byte) lock-free `FrameBufferPool`
//! - Cross-platform `ScreenCapturer` trait with multi-monitor display enumeration
//! - High-speed SIMD tile damage detection and dirty rectangle tracking
//! - Deterministic headless `MockCapturer` for automated testing

#![warn(missing_docs)]

pub mod backends;
pub mod buffer;
pub mod capturer;
pub mod damage;
pub mod rect;

pub use backends::MockCapturer;
pub use buffer::{FrameBuffer, FrameBufferPool, FrameGuard, PixelFormat, CACHELINE_ALIGN_BYTES};
pub use capturer::{CaptureError, CapturedFrame, ScreenCapturer};
pub use damage::{DamageScanner, TILE_SIZE};
pub use rect::{DisplayInfo, DisplayTopology, Rect};
