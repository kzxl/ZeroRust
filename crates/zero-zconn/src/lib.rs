//! # ZeroZConn
//!
//! Sovereign ultra-low-latency remote desktop orchestrator and runtime for ZeroRust:
//! - High-performance `ZConnHost` capture-encode-transmit pipeline
//! - Lightweight `ZConnClient` datagram-ingest-decode-present pipeline
//! - Quality presets: `UltraLowLatency`, `Balanced`, `HighFidelity`
//! - C-FFI exports for ZeroPlatform (.NET 8/9 C# P/Invoke) presentation surfaces

#![warn(missing_docs)]

pub mod client;
pub mod config;
pub mod ffi;
pub mod host;
pub mod session;

pub use client::ZConnClient;
pub use config::{ClientConfig, HostConfig, QualityPreset};
pub use host::ZConnHost;
pub use session::{DeviceInfo, SessionMetrics, SessionState};

#[cfg(test)]
mod tests {
    use super::*;
    use zero_capture::MockCapturer;
    use zero_input::MockInjector;

    #[test]
    fn test_host_to_client_end_to_end_pipeline() {
        let width = 320;
        let height = 240;

        // 1. Instantiate Host with Mock Capturer and Mock Injector
        let capturer = Box::new(MockCapturer::new(width, height));
        let injector = Box::new(MockInjector::new());
        let mut host = ZConnHost::new(HostConfig::default(), capturer, injector);

        // 2. Instantiate Client
        let mut client = ZConnClient::new(ClientConfig::default(), width, height);

        // 3. Host executes a frame step (produces video tile datagrams)
        let packets = host.process_frame_step(16).expect("Host frame step");
        assert!(!packets.is_empty(), "Should produce tile datagrams");

        // 4. Client ingests all packets
        for pkt_bytes in &packets {
            client.handle_incoming_packet(pkt_bytes);
        }

        // 5. Verify client rendered surface has received non-zero pixel data
        let surface = client.render_surface();
        assert_eq!(surface.len(), (width as usize) * 4 * (height as usize));
        let non_zero_pixels = surface.iter().filter(|&&b| b != 0).count();
        assert!(
            non_zero_pixels > 0,
            "Client surface must contain rendered desktop data"
        );

        // 6. Client dispatches a mouse movement back to host
        let mouse_ev = zero_input::MouseEvent::MoveAbsolute {
            norm_x: 0.5,
            norm_y: 0.5,
            display_id: 0,
        };
        let input_packet = client.build_mouse_packet(&mouse_ev, 20);
        host.handle_incoming_packet(&input_packet, 20);

        assert_eq!(host.metrics().frames_processed, 1);
        assert_eq!(client.metrics().frames_processed, 1);
    }
}
