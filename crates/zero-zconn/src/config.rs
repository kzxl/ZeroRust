//! Host and client configuration parameters and quality presets.

use std::net::SocketAddr;

/// Streaming quality and performance preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityPreset {
    /// Ultra-low latency mode (<10ms) prioritizing responsive gaming / CAD interaction.
    UltraLowLatency,
    /// Balanced everyday mode prioritizing text clarity with smooth 60 FPS motion.
    #[default]
    Balanced,
    /// High-fidelity mode prioritizing 100% pixel-perfect lossless text rendering.
    HighFidelity,
}

/// Host daemon configuration.
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// Local UDP socket bind address.
    pub bind_addr: SocketAddr,
    /// Target display monitor index to stream (default: 0).
    pub display_id: usize,
    /// Maximum target frame rate in Hertz.
    pub max_fps: u8,
    /// Quality preset.
    pub preset: QualityPreset,
    /// 256-bit symmetric encryption key (all zeroes disables E2EE for local LAN).
    pub encryption_key: [u8; 32],
    /// Optional session PIN / password for client authentication.
    pub password: String,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:21118".parse().unwrap(),
            display_id: 0,
            max_fps: 60,
            preset: QualityPreset::Balanced,
            encryption_key: [0u8; 32],
            password: String::new(),
        }
    }
}

/// Client renderer and control configuration.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Target remote host UDP address.
    pub target_addr: SocketAddr,
    /// Quality preset.
    pub preset: QualityPreset,
    /// 256-bit symmetric encryption key matching the host.
    pub encryption_key: [u8; 32],
    /// Remote host PIN / password.
    pub password: String,
    /// Enable bi-directional clipboard synchronization.
    pub enable_clipboard: bool,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            target_addr: "127.0.0.1:21118".parse().unwrap(),
            preset: QualityPreset::Balanced,
            encryption_key: [0u8; 32],
            password: String::new(),
            enable_clipboard: true,
        }
    }
}
