//! ZConn Sovereign Remote Desktop Client Engine.

use zero_codec::{
    apply_tile_xor, decompress_tile_rle, TileFragmentHeader, TileHeader, TileKind, TILE_RAW_BYTES,
};
use zero_input::{ClipboardSyncManager, KeyboardEvent, MouseEvent};
use zero_tunnel::{
    CryptoSession, FrameJitterBuffer, JitterAction, ZProtoChannel, ZProtoPacket, ZProtoPacketType,
};

use crate::config::ClientConfig;
use crate::session::{SessionMetrics, SessionState};

/// Remote desktop client engine managing packet ingest, tile decoding, and presentation.
pub struct ZConnClient {
    config: ClientConfig,
    state: SessionState,
    width: u32,
    height: u32,
    stride: usize,
    render_surface: Vec<u8>,
    jitter_buffer: FrameJitterBuffer,
    clipboard_mgr: ClipboardSyncManager,
    crypto: CryptoSession,
    metrics: SessionMetrics,
    input_sequence: u32,
    last_rendered_frame: u32,
}

impl ZConnClient {
    /// Creates a new client engine for a given display resolution.
    pub fn new(config: ClientConfig, width: u32, height: u32) -> Self {
        let stride = (width as usize) * 4;
        let render_surface = vec![0u8; stride * (height as usize)];
        let crypto = CryptoSession::new(config.encryption_key);

        Self {
            config,
            state: SessionState::Connected,
            width,
            height,
            stride,
            render_surface,
            jitter_buffer: FrameJitterBuffer::new(),
            clipboard_mgr: ClipboardSyncManager::new(),
            crypto,
            metrics: SessionMetrics::default(),
            input_sequence: 0,
            last_rendered_frame: 0,
        }
    }

    /// Ingests a raw UDP datagram received from the host.
    pub fn handle_incoming_packet(&mut self, packet_bytes: &[u8]) {
        self.metrics.bytes_received += packet_bytes.len() as u64;

        if let Some(pkt) = ZProtoPacket::from_bytes(packet_bytes) {
            if pkt.header.packet_type == ZProtoPacketType::TileData as u8 {
                if let Some(frag_hdr) = TileFragmentHeader::from_bytes(&pkt.payload) {
                    let frag_payload = &pkt.payload[TileFragmentHeader::SIZE..];
                    let action = self.jitter_buffer.push_fragment(
                        frag_hdr.frame_id,
                        frag_hdr.tile_index,
                        frag_hdr.chunk_index,
                        frag_hdr.total_chunks,
                        frag_payload.to_vec(),
                    );

                    if let JitterAction::TileComplete {
                        frame_id, payload, ..
                    } = action
                    {
                        self.decode_and_apply_tile(&payload);
                        self.metrics.tiles_processed += 1;
                        if frame_id > self.last_rendered_frame {
                            self.last_rendered_frame = frame_id;
                            self.metrics.frames_processed += 1;
                        }
                        self.jitter_buffer.mark_frame_rendered(frame_id);
                    }
                }
            }
        }
    }

    /// Decodes a reassembled tile payload and writes it directly into the render surface.
    fn decode_and_apply_tile(&mut self, tile_bytes: &[u8]) {
        if let Some(header) = TileHeader::from_bytes(tile_bytes) {
            let px = (header.tile_x as usize) * 32;
            let py = (header.tile_y as usize) * 32;
            let byte_offset = py * self.stride + px * 4;

            if byte_offset + (31 * self.stride) + 128 > self.render_surface.len() {
                return;
            }

            let payload = &tile_bytes[TileHeader::SIZE..];
            let kind = TileKind::from_u8(header.kind).unwrap_or(TileKind::Unchanged);

            match kind {
                TileKind::SolidColor => {
                    if payload.len() >= 4 {
                        let color_bytes = [payload[0], payload[1], payload[2], payload[3]];
                        for row in 0..32 {
                            let row_start = byte_offset + row * self.stride;
                            for col in 0..32 {
                                let p = row_start + col * 4;
                                self.render_surface[p..p + 4].copy_from_slice(&color_bytes);
                            }
                        }
                    }
                }
                TileKind::DeltaRle | TileKind::FullIntra => {
                    let mut delta = vec![0u8; TILE_RAW_BYTES];
                    if decompress_tile_rle(payload, &mut delta).is_ok() {
                        unsafe {
                            let out_ptr = self.render_surface.as_mut_ptr().add(byte_offset);
                            apply_tile_xor(out_ptr, delta.as_ptr(), out_ptr, self.stride);
                        }
                    }
                }
                TileKind::DeltaRaw => {
                    if payload.len() >= TILE_RAW_BYTES {
                        unsafe {
                            let out_ptr = self.render_surface.as_mut_ptr().add(byte_offset);
                            apply_tile_xor(out_ptr, payload.as_ptr(), out_ptr, self.stride);
                        }
                    }
                }
                TileKind::Unchanged => {}
            }
        }
    }

    /// Formats a mouse event into a UDP datagram packet ready for transmission to the host.
    pub fn build_mouse_packet(&mut self, event: &MouseEvent, now_ms: u32) -> Vec<u8> {
        self.input_sequence = self.input_sequence.wrapping_add(1);
        let payload = event.to_bytes();
        let pkt = ZProtoPacket::new(
            ZProtoChannel::Input,
            0x5A43_0001,
            self.input_sequence,
            now_ms,
            ZProtoPacketType::InputEvent,
            payload,
        );
        let bytes = pkt.to_bytes();
        self.metrics.bytes_transmitted += bytes.len() as u64;
        bytes
    }

    /// Formats a keyboard event into a UDP datagram packet ready for transmission to the host.
    pub fn build_keyboard_packet(&mut self, event: &KeyboardEvent, now_ms: u32) -> Vec<u8> {
        self.input_sequence = self.input_sequence.wrapping_add(1);
        let payload = event.to_bytes().to_vec();
        let pkt = ZProtoPacket::new(
            ZProtoChannel::Input,
            0x5A43_0001,
            self.input_sequence,
            now_ms,
            ZProtoPacketType::InputEvent,
            payload,
        );
        let bytes = pkt.to_bytes();
        self.metrics.bytes_transmitted += bytes.len() as u64;
        bytes
    }

    /// Returns a slice to the current rendered frame pixel buffer (BGRA 32bpp).
    pub fn render_surface(&self) -> &[u8] {
        &self.render_surface
    }

    /// Display width in physical pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Display height in physical pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Stride in bytes per scanline.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns session streaming metrics.
    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }

    /// Returns client configuration.
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Returns session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Returns mutable reference to clipboard manager.
    pub fn clipboard_mgr_mut(&mut self) -> &mut ClipboardSyncManager {
        &mut self.clipboard_mgr
    }

    /// Returns mutable reference to crypto session.
    pub fn crypto_mut(&mut self) -> &mut CryptoSession {
        &mut self.crypto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_initialization_and_input_dispatch() {
        let mut client = ZConnClient::new(ClientConfig::default(), 640, 480);
        assert_eq!(client.width(), 640);
        assert_eq!(client.height(), 480);
        assert_eq!(client.render_surface().len(), 640 * 4 * 480);

        let mouse_ev = MouseEvent::MoveRelative {
            delta_x: 10,
            delta_y: -5,
        };
        let pkt_bytes = client.build_mouse_packet(&mouse_ev, 100);
        assert!(!pkt_bytes.is_empty());
    }
}
