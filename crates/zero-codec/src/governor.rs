//! Adaptive bandwidth governor and progressive intra-refresh coordinator.

/// Dynamic quality and frame rate controller driven by network feedback.
#[derive(Debug, Clone)]
pub struct BandwidthGovernor {
    target_fps: u8,
    quantization_bits: u8,
    rtt_ms: f32,
    packet_loss_rate: f32,
    bitrate_estimate_bps: u64,
}

impl Default for BandwidthGovernor {
    fn default() -> Self {
        Self {
            target_fps: 60,
            quantization_bits: 0,
            rtt_ms: 10.0,
            packet_loss_rate: 0.0,
            bitrate_estimate_bps: 10_000_000, // 10 Mbps default baseline
        }
    }
}

impl BandwidthGovernor {
    /// Creates a new bandwidth governor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates network telemetry and recalculates optimal video parameters.
    pub fn update_network_stats(&mut self, rtt_ms: f32, loss_rate: f32) {
        self.rtt_ms = rtt_ms;
        self.packet_loss_rate = loss_rate;

        // Congestion response
        if loss_rate > 0.05 || rtt_ms > 120.0 {
            // Drop target frame rate, increase quantization
            self.target_fps = self.target_fps.saturating_sub(15).max(15);
            self.quantization_bits = (self.quantization_bits + 1).min(3);
        } else if loss_rate < 0.005 && rtt_ms < 40.0 {
            // High-speed link: scale up to 60 FPS, lossless
            self.target_fps = (self.target_fps + 15).min(60);
            self.quantization_bits = self.quantization_bits.saturating_sub(1);
        }
    }

    /// Returns active target frame rate.
    pub fn target_fps(&self) -> u8 {
        self.target_fps
    }

    /// Frame duration budget in milliseconds.
    pub fn frame_interval_ms(&self) -> u64 {
        1000 / (self.target_fps as u64).max(1)
    }

    /// Returns active quantization truncation bits (0 = lossless).
    pub fn quantization_bits(&self) -> u8 {
        self.quantization_bits
    }

    /// Estimated link throughput in bits per second.
    pub fn bitrate_estimate_bps(&self) -> u64 {
        self.bitrate_estimate_bps
    }
}

/// Progressive rolling intra-refresh manager.
/// Updates 5% of tiles per frame as keyframes to heal packet loss without network bursts.
#[derive(Debug, Clone)]
pub struct RollingIntraManager {
    total_tiles: usize,
    chunk_size: usize,
}

impl RollingIntraManager {
    /// Creates a rolling intra manager for a total tile count.
    pub fn new(total_tiles: usize) -> Self {
        let chunk_size = total_tiles.div_ceil(20); // 5% per cycle
        Self {
            total_tiles,
            chunk_size: chunk_size.max(1),
        }
    }

    /// Determines if a specific tile should be sent as an intra keyframe in `frame_id`.
    pub fn is_intra_tile(&self, frame_id: u64, tile_index: usize) -> bool {
        if self.total_tiles == 0 {
            return false;
        }
        let cycle_step = (frame_id as usize) % 20;
        let start = cycle_step * self.chunk_size;
        let end = (start + self.chunk_size).min(self.total_tiles);

        tile_index >= start && tile_index < end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_governor() {
        let mut gov = BandwidthGovernor::new();
        assert_eq!(gov.target_fps(), 60);

        // Simulate high loss
        gov.update_network_stats(150.0, 0.08);
        assert!(gov.target_fps() < 60);
        assert!(gov.quantization_bits() > 0);

        // Simulate recovery
        for _ in 0..5 {
            gov.update_network_stats(15.0, 0.001);
        }
        assert_eq!(gov.target_fps(), 60);
        assert_eq!(gov.quantization_bits(), 0);
    }

    #[test]
    fn test_rolling_intra_manager() {
        let mgr = RollingIntraManager::new(100);
        // Step 0 covers 0..5
        assert!(mgr.is_intra_tile(0, 0));
        assert!(mgr.is_intra_tile(0, 4));
        assert!(!mgr.is_intra_tile(0, 10));

        // Step 1 covers 5..10
        assert!(mgr.is_intra_tile(1, 5));
        assert!(!mgr.is_intra_tile(1, 0));
    }
}
