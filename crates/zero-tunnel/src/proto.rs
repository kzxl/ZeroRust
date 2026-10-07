//! ZProto multiplexed low-latency datagram framing protocol.

/// Magic 16-bit word identifying ZProto packets (`0x5A43` = ASCII "ZC").
pub const ZPROTO_MAGIC: u16 = 0x5A43;

/// Current protocol wire version.
pub const ZPROTO_VERSION: u8 = 1;

/// Multiplexed logical channels within a single UDP socket session.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZProtoChannel {
    /// Reliable control channel (handshake, capability negotiation, session keys).
    Control = 0,
    /// High-priority input event channel (mouse, keyboard).
    Input = 1,
    /// High-throughput video tile channel (loss-tolerant, drop stale frames).
    Video = 2,
    /// Real-time audio stream channel.
    Audio = 3,
    /// Reliable bulk data transfer channel (clipboard, file transfer).
    Bulk = 4,
}

impl ZProtoChannel {
    /// Decodes channel from wire byte.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Control),
            1 => Some(Self::Input),
            2 => Some(Self::Video),
            3 => Some(Self::Audio),
            4 => Some(Self::Bulk),
            _ => None,
        }
    }
}

/// Message type discriminator.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZProtoPacketType {
    /// Session initiation request.
    Handshake = 0x01,
    /// Session initiation acknowledgment with public key.
    HandshakeAck = 0x02,
    /// Heartbeat ping.
    Ping = 0x03,
    /// Heartbeat pong.
    Pong = 0x04,
    /// Video screen tile payload fragment.
    TileData = 0x10,
    /// Synthetic input injection payload.
    InputEvent = 0x20,
    /// Clipboard synchronization payload.
    Clipboard = 0x30,
}

impl ZProtoPacketType {
    /// Decodes packet type from wire byte.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Handshake),
            0x02 => Some(Self::HandshakeAck),
            0x03 => Some(Self::Ping),
            0x04 => Some(Self::Pong),
            0x10 => Some(Self::TileData),
            0x20 => Some(Self::InputEvent),
            0x30 => Some(Self::Clipboard),
            _ => None,
        }
    }
}

/// Binary wire header accompanying every ZProto datagram (20 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZProtoHeader {
    /// Magic identifier (`0x5A43`).
    pub magic: u16,
    /// Wire protocol version (`1`).
    pub version: u8,
    /// Multiplexed logical channel.
    pub channel: u8,
    /// Unique 32-bit session identifier.
    pub session_id: u32,
    /// Monotonically increasing sequence number.
    pub sequence: u32,
    /// Millisecond session timestamp.
    pub timestamp_ms: u32,
    /// Packet type.
    pub packet_type: u8,
    /// Reserved flags (bit 0 = encrypted).
    pub flags: u8,
    /// Payload length in bytes following this header.
    pub payload_len: u16,
}

impl ZProtoHeader {
    /// Size of binary header in bytes (20 bytes).
    pub const SIZE: usize = 20;

    /// Serializes header into byte array.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut b = [0u8; Self::SIZE];
        b[0..2].copy_from_slice(&self.magic.to_le_bytes());
        b[2] = self.version;
        b[3] = self.channel;
        b[4..8].copy_from_slice(&self.session_id.to_le_bytes());
        b[8..12].copy_from_slice(&self.sequence.to_le_bytes());
        b[12..16].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        b[16] = self.packet_type;
        b[17] = self.flags;
        b[18..20].copy_from_slice(&self.payload_len.to_le_bytes());
        b
    }

    /// Deserializes header from byte slice.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < Self::SIZE {
            return None;
        }
        let magic = u16::from_le_bytes([slice[0], slice[1]]);
        if magic != ZPROTO_MAGIC {
            return None;
        }
        let version = slice[2];
        let channel = slice[3];
        let session_id = u32::from_le_bytes([slice[4], slice[5], slice[6], slice[7]]);
        let sequence = u32::from_le_bytes([slice[8], slice[9], slice[10], slice[11]]);
        let timestamp_ms = u32::from_le_bytes([slice[12], slice[13], slice[14], slice[15]]);
        let packet_type = slice[16];
        let flags = slice[17];
        let payload_len = u16::from_le_bytes([slice[18], slice[19]]);

        Some(Self {
            magic,
            version,
            channel,
            session_id,
            sequence,
            timestamp_ms,
            packet_type,
            flags,
            payload_len,
        })
    }
}

/// High-level datagram packet with header and payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZProtoPacket {
    /// Packet header.
    pub header: ZProtoHeader,
    /// Raw or encrypted payload bytes.
    pub payload: Vec<u8>,
}

impl ZProtoPacket {
    /// Creates a new datagram packet.
    pub fn new(
        channel: ZProtoChannel,
        session_id: u32,
        sequence: u32,
        timestamp_ms: u32,
        packet_type: ZProtoPacketType,
        payload: Vec<u8>,
    ) -> Self {
        let payload_len = payload.len() as u16;
        let header = ZProtoHeader {
            magic: ZPROTO_MAGIC,
            version: ZPROTO_VERSION,
            channel: channel as u8,
            session_id,
            sequence,
            timestamp_ms,
            packet_type: packet_type as u8,
            flags: 0,
            payload_len,
        };
        Self { header, payload }
    }

    /// Serializes entire packet (header + payload) into wire bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(ZProtoHeader::SIZE + self.payload.len());
        b.extend_from_slice(&self.header.to_bytes());
        b.extend_from_slice(&self.payload);
        b
    }

    /// Deserializes entire packet from wire bytes.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        let header = ZProtoHeader::from_bytes(slice)?;
        let expected_len = ZProtoHeader::SIZE + header.payload_len as usize;
        if slice.len() < expected_len {
            return None;
        }
        let payload = slice[ZProtoHeader::SIZE..expected_len].to_vec();
        Some(Self { header, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zproto_packet_roundtrip() {
        let payload = b"ZConn Ping Data 12345".to_vec();
        let pkt = ZProtoPacket::new(
            ZProtoChannel::Control,
            0xAABBCCDD,
            105,
            5000,
            ZProtoPacketType::Ping,
            payload.clone(),
        );

        let bytes = pkt.to_bytes();
        let parsed = ZProtoPacket::from_bytes(&bytes).expect("Parse ZProto packet");

        assert_eq!(parsed.header.magic, ZPROTO_MAGIC);
        assert_eq!(parsed.header.channel, ZProtoChannel::Control as u8);
        assert_eq!(parsed.header.session_id, 0xAABBCCDD);
        assert_eq!(parsed.header.sequence, 105);
        assert_eq!(parsed.header.packet_type, ZProtoPacketType::Ping as u8);
        assert_eq!(parsed.payload, payload);
    }
}
