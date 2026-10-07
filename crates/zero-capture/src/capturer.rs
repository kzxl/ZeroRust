//! Screen capturer trait definitions and capture error types.

use crate::buffer::{FrameBufferPool, FrameGuard};
use crate::rect::{DisplayInfo, Rect};

/// Errors encountered during screen capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// Screen capture device lost (e.g. UAC secure desktop, resolution change).
    DeviceLost,
    /// Capture request timed out waiting for the next vertical blank / frame.
    Timeout,
    /// The pre-allocated frame buffer pool is exhausted.
    BufferPoolExhausted,
    /// Output monitor is disconnected or invalid.
    MonitorDisconnected,
    /// Operating system platform error.
    Platform(String),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceLost => write!(f, "Capture device lost"),
            Self::Timeout => write!(f, "Capture timeout"),
            Self::BufferPoolExhausted => write!(f, "Frame buffer pool exhausted"),
            Self::MonitorDisconnected => write!(f, "Monitor disconnected"),
            Self::Platform(msg) => write!(f, "Platform capture error: {}", msg),
        }
    }
}

impl std::error::Error for CaptureError {}

/// Captured frame packet containing the raw pixel buffer and damage metadata.
pub struct CapturedFrame<'a> {
    /// RAII guard wrapping the acquired frame buffer.
    pub buffer: FrameGuard<'a>,
    /// List of dirty / modified rectangular regions.
    pub damage_rects: Vec<Rect>,
    /// Indicates whether this frame must be treated as a keyframe (full intra).
    pub is_keyframe: bool,
    /// Monotonic timestamp in nanoseconds when the frame was acquired.
    pub timestamp_ns: u64,
}

/// Abstract cross-platform screen capturer interface.
pub trait ScreenCapturer: Send {
    /// Captures the next display frame into a buffer from `pool`.
    fn capture_frame<'a>(
        &mut self,
        pool: &'a FrameBufferPool,
        timeout_ms: u32,
    ) -> Result<CapturedFrame<'a>, CaptureError>;

    /// Queries the display information for the active monitor.
    fn display_info(&self) -> DisplayInfo;

    /// Returns the active monitor index.
    fn display_id(&self) -> usize;
}
