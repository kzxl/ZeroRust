//! Session lifecycle, authentication metadata, and real-time streaming telemetry.

/// Lifecycle states of a remote desktop connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SessionState {
    /// No connection active.
    #[default]
    Disconnected,
    /// Handshake dispatched, waiting for authentication challenge.
    Authenticating,
    /// Authenticated and connected, preparing video pipeline.
    Connected,
    /// Actively capturing, encoding, and streaming screen frames.
    Streaming,
    /// Connection encountered fatal error or was closed.
    Terminated,
}

/// Unique hardware and software device identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Sovereign 64-bit hardware device identifier.
    pub device_id: u64,
    /// Computer hostname.
    pub hostname: String,
    /// Operating system platform ("Windows", "Linux", etc.).
    pub os_type: String,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            device_id: 0x5A43_0001_0002_0003,
            hostname: "ZConn-Host".to_string(),
            os_type: std::env::consts::OS.to_string(),
        }
    }
}

/// Real-time session streaming telemetry metrics.
#[derive(Debug, Clone, Default)]
pub struct SessionMetrics {
    /// Total frames captured and processed.
    pub frames_processed: u64,
    /// Total tiles decoded or encoded.
    pub tiles_processed: u64,
    /// Total network bytes transmitted.
    pub bytes_transmitted: u64,
    /// Total network bytes received.
    pub bytes_received: u64,
    /// Measured round-trip network latency in milliseconds.
    pub rtt_ms: f32,
    /// Current measured frame rate (FPS).
    pub current_fps: f32,
    /// Estimated packet loss rate ($0.0 \dots 1.0$).
    pub packet_loss_rate: f32,
}
