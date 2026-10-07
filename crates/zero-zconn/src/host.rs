//! ZConn Sovereign Remote Desktop Host Daemon.

use std::net::SocketAddr;
use zero_capture::{CaptureError, FrameBufferPool, PixelFormat, ScreenCapturer, TILE_SIZE};
use zero_codec::{
    compress_tile_rle, compute_tile_xor, BandwidthGovernor, RollingIntraManager, TileHeader,
    TileKind, TilePacketizer, TILE_RAW_BYTES,
};
use zero_input::{InputInjector, KeyboardEvent, MouseEvent, SafetyGuard};
use zero_tunnel::{CryptoSession, ZProtoChannel, ZProtoPacket, ZProtoPacketType};

use crate::config::HostConfig;
use crate::session::{DeviceInfo, SessionMetrics, SessionState};

/// Remote desktop host service orchestrating capture, encoding, and input dispatch.
pub struct ZConnHost {
    config: HostConfig,
    device_info: DeviceInfo,
    state: SessionState,
    capturer: Box<dyn ScreenCapturer>,
    injector: Box<dyn InputInjector>,
    buffer_pool: FrameBufferPool,
    prev_frame: Vec<u8>,
    rolling_intra: RollingIntraManager,
    governor: BandwidthGovernor,
    crypto: CryptoSession,
    safety_guard: SafetyGuard,
    metrics: SessionMetrics,
    frame_sequence: u32,
    client_addr: Option<SocketAddr>,
}

impl ZConnHost {
    /// Creates a new host instance with concrete capture and input engines.
    pub fn new(
        config: HostConfig,
        capturer: Box<dyn ScreenCapturer>,
        injector: Box<dyn InputInjector>,
    ) -> Self {
        let disp = capturer.display_info();
        let buffer_pool = FrameBufferPool::new(3, disp.width, disp.height, PixelFormat::Bgra8888);
        let stride = (disp.width as usize) * 4;
        let prev_frame = vec![0u8; stride * (disp.height as usize)];

        let total_tiles = disp.width.div_ceil(TILE_SIZE) * disp.height.div_ceil(TILE_SIZE);
        let rolling_intra = RollingIntraManager::new(total_tiles as usize);
        let crypto = CryptoSession::new(config.encryption_key);

        Self {
            config,
            device_info: DeviceInfo::default(),
            state: SessionState::Disconnected,
            capturer,
            injector,
            buffer_pool,
            prev_frame,
            rolling_intra,
            governor: BandwidthGovernor::new(),
            crypto,
            safety_guard: SafetyGuard::new(2000),
            metrics: SessionMetrics::default(),
            frame_sequence: 0,
            client_addr: None,
        }
    }

    /// Sets the active client address for streaming.
    pub fn connect_client(&mut self, addr: SocketAddr) {
        self.client_addr = Some(addr);
        self.state = SessionState::Streaming;
    }

    /// Disconnects the active client and releases any held keys.
    pub fn disconnect_client(&mut self) {
        let releases = self.safety_guard.release_all_keys();
        for ev in releases {
            let _ = self.injector.inject_keyboard(&ev);
        }
        self.client_addr = None;
        self.state = SessionState::Disconnected;
    }

    /// Executes a single capture and encode iteration.
    /// Returns vector of serialized UDP packets ready for transmission.
    pub fn process_frame_step(&mut self, now_ms: u32) -> Result<Vec<Vec<u8>>, CaptureError> {
        let captured = self.capturer.capture_frame(&self.buffer_pool, 50)?;
        self.frame_sequence = self.frame_sequence.wrapping_add(1);
        let frame_id = self.frame_sequence;
        let stride = captured.buffer.stride;
        let w = captured.buffer.width;
        let h = captured.buffer.height;
        let cur_slice = captured.buffer.as_slice();

        let mut outgoing_packets = Vec::new();
        let mut delta_scratch = vec![0u8; TILE_RAW_BYTES];
        let mut rle_scratch = Vec::with_capacity(TILE_RAW_BYTES);

        let tiles_x = w.div_ceil(TILE_SIZE);
        let tiles_y = h.div_ceil(TILE_SIZE);

        for ty in 0..tiles_y {
            let py = ty * TILE_SIZE;
            for tx in 0..tiles_x {
                let px = tx * TILE_SIZE;
                let tile_idx = (ty * tiles_x + tx) as usize;
                let is_intra = captured.is_keyframe
                    || self.rolling_intra.is_intra_tile(frame_id as u64, tile_idx);

                let byte_offset = (py as usize) * stride + (px as usize) * 4;
                if byte_offset + (31 * stride) + 128 > cur_slice.len() {
                    continue; // Skip boundary incomplete tiles for brevity
                }

                unsafe {
                    let cur_ptr = cur_slice.as_ptr().add(byte_offset);
                    let ref_ptr = self.prev_frame.as_ptr().add(byte_offset);

                    let analysis =
                        compute_tile_xor(cur_ptr, ref_ptr, delta_scratch.as_mut_ptr(), stride);

                    if !is_intra && analysis.is_identical {
                        continue; // 0-byte unchanged tile: completely skipped!
                    }

                    rle_scratch.clear();

                    let (kind, payload_slice) = if analysis.is_solid {
                        (
                            TileKind::SolidColor,
                            &analysis.solid_color.to_le_bytes()[..],
                        )
                    } else if is_intra {
                        compress_tile_rle(&delta_scratch, &mut rle_scratch);
                        (TileKind::FullIntra, rle_scratch.as_slice())
                    } else {
                        compress_tile_rle(&delta_scratch, &mut rle_scratch);
                        (TileKind::DeltaRle, rle_scratch.as_slice())
                    };

                    let header = TileHeader {
                        tile_x: tx as u16,
                        tile_y: ty as u16,
                        kind: kind as u8,
                        flags: if is_intra { 0x01 } else { 0x00 },
                        payload_len: payload_slice.len() as u16,
                    };

                    let mut tile_packet_data =
                        Vec::with_capacity(TileHeader::SIZE + payload_slice.len());
                    tile_packet_data.extend_from_slice(&header.to_bytes());
                    tile_packet_data.extend_from_slice(payload_slice);

                    // Fragment into MTU 1160 datagrams
                    let frags =
                        TilePacketizer::packetize(frame_id, tile_idx as u16, &tile_packet_data);
                    for frag_data in frags {
                        let pkt = ZProtoPacket::new(
                            ZProtoChannel::Video,
                            0x5A43_0001,
                            frame_id,
                            now_ms,
                            ZProtoPacketType::TileData,
                            frag_data,
                        );
                        let bytes = pkt.to_bytes();
                        self.metrics.bytes_transmitted += bytes.len() as u64;
                        outgoing_packets.push(bytes);
                    }
                }
            }
        }

        // Cache reference frame
        let copy_len = self.prev_frame.len().min(cur_slice.len());
        self.prev_frame[..copy_len].copy_from_slice(&cur_slice[..copy_len]);
        self.metrics.frames_processed += 1;

        Ok(outgoing_packets)
    }

    /// Handles an incoming UDP datagram from the client.
    pub fn handle_incoming_packet(&mut self, packet_bytes: &[u8], now_ms: u64) {
        self.metrics.bytes_received += packet_bytes.len() as u64;
        if let Some(pkt) = ZProtoPacket::from_bytes(packet_bytes) {
            match pkt.header.packet_type {
                0x20 => {
                    // InputEvent
                    if let Some(mouse) = MouseEvent::from_bytes(&pkt.payload) {
                        let _ = self.injector.inject_mouse(&mouse);
                    } else if let Some(key) = KeyboardEvent::from_bytes(&pkt.payload) {
                        self.safety_guard.register_key(&key, now_ms);
                        let _ = self.injector.inject_keyboard(&key);
                    }
                }
                0x03 => {
                    // Ping
                    // In real daemon, responds with Pong
                }
                _ => {}
            }
        }
    }

    /// Returns session metrics.
    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }

    /// Returns session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Returns host configuration.
    pub fn config(&self) -> &HostConfig {
        &self.config
    }

    /// Returns host device info.
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Returns bandwidth governor reference.
    pub fn governor(&self) -> &BandwidthGovernor {
        &self.governor
    }

    /// Returns mutable reference to crypto session.
    pub fn crypto_mut(&mut self) -> &mut CryptoSession {
        &mut self.crypto
    }
}
