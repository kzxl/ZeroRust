//! Micro-XRCE-DDS transport framing and submessage serialization.

use zero_core::error::{ZeroError, ZeroResult};

/// Submessage identifiers defined by OMG Micro-XRCE-DDS standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SubmessageId {
    /// Status response from agent (0x00).
    Status = 0x00,
    /// Create entity in agent (0x01).
    Create = 0x01,
    /// Query entity status (0x02).
    GetInfo = 0x02,
    /// Delete entity (0x03).
    Delete = 0x03,
    /// Status Agent announcement (0x05).
    StatusAgent = 0x05,
    /// Write topic payload data to DDS (0x07).
    WriteData = 0x07,
    /// Read topic data from DDS (0x08).
    ReadData = 0x08,
    /// Acknowledgment / Negative acknowledgment for reliable stream (0x0A).
    AckNack = 0x0A,
    /// Heartbeat message (0x0B).
    Heartbeat = 0x0B,
}

impl TryFrom<u8> for SubmessageId {
    type Error = ZeroError;

    fn try_from(val: u8) -> Result<Self, Self::Error> {
        match val {
            0x00 => Ok(Self::Status),
            0x01 => Ok(Self::Create),
            0x02 => Ok(Self::GetInfo),
            0x03 => Ok(Self::Delete),
            0x05 => Ok(Self::StatusAgent),
            0x07 => Ok(Self::WriteData),
            0x08 => Ok(Self::ReadData),
            0x0A => Ok(Self::AckNack),
            0x0B => Ok(Self::Heartbeat),
            _ => Err(ZeroError::InvalidArgument),
        }
    }
}

/// Standard Micro-XRCE-DDS message header (8 bytes with Client Key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageHeader {
    /// Session identifier (0x80..0xFF for client sessions).
    pub session_id: u8,
    /// Stream identifier (0x01..0x7F for best-effort, 0x80..0xFF for reliable).
    pub stream_id: u8,
    /// Sequence number for packet ordering.
    pub sequence_nr: u16,
    /// 4-byte unique client identifier key.
    pub client_key: [u8; 4],
}

impl MessageHeader {
    /// Serializes XRCE header into buffer (8 bytes).
    pub fn serialize(&self, buf: &mut [u8]) -> ZeroResult<usize> {
        if buf.len() < 8 {
            return Err(ZeroError::BufferOverflow);
        }
        buf[0] = self.session_id;
        buf[1] = self.stream_id;
        let seq_bytes = self.sequence_nr.to_le_bytes();
        buf[2] = seq_bytes[0];
        buf[3] = seq_bytes[1];
        buf[4..8].copy_from_slice(&self.client_key);
        Ok(8)
    }

    /// Deserializes XRCE header from buffer.
    pub fn deserialize(buf: &[u8]) -> ZeroResult<Self> {
        if buf.len() < 8 {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let session_id = buf[0];
        let stream_id = buf[1];
        let sequence_nr = u16::from_le_bytes([buf[2], buf[3]]);
        let mut client_key = [0u8; 4];
        client_key.copy_from_slice(&buf[4..8]);
        Ok(Self {
            session_id,
            stream_id,
            sequence_nr,
            client_key,
        })
    }
}

/// Micro-XRCE-DDS Submessage Header (4 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmessageHeader {
    /// Submessage kind.
    pub submsg_id: SubmessageId,
    /// Flags (bit 0 = Little Endian).
    pub flags: u8,
    /// Payload length excluding this 4-byte header.
    pub length: u16,
}

impl SubmessageHeader {
    /// Serializes submessage header into buffer (4 bytes).
    pub fn serialize(&self, buf: &mut [u8]) -> ZeroResult<usize> {
        if buf.len() < 4 {
            return Err(ZeroError::BufferOverflow);
        }
        buf[0] = self.submsg_id as u8;
        buf[1] = self.flags;
        let len_bytes = self.length.to_le_bytes();
        buf[2] = len_bytes[0];
        buf[3] = len_bytes[1];
        Ok(4)
    }

    /// Deserializes submessage header from buffer.
    pub fn deserialize(buf: &[u8]) -> ZeroResult<Self> {
        if buf.len() < 4 {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let submsg_id = SubmessageId::try_from(buf[0])?;
        let flags = buf[1];
        let length = u16::from_le_bytes([buf[2], buf[3]]);
        Ok(Self {
            submsg_id,
            flags,
            length,
        })
    }
}

/// Helper to assemble a complete `WRITE_DATA` publish message.
pub struct WriteDataPayload;

impl WriteDataPayload {
    /// Packs a `WRITE_DATA` submessage wrapping a serialized topic payload.
    ///
    /// Total bytes = 8 (MessageHeader) + 4 (SubmessageHeader) + 4 (Request/ObjectId) + payload_len.
    pub fn pack_frame(
        header: &MessageHeader,
        object_id: u16,
        topic_payload: &[u8],
        out_buf: &mut [u8],
    ) -> ZeroResult<usize> {
        let total_len = 8 + 4 + 4 + topic_payload.len();
        if out_buf.len() < total_len {
            return Err(ZeroError::BufferOverflow);
        }

        // 1. Message Header (8 bytes)
        header.serialize(&mut out_buf[0..8])?;

        // 2. Submessage Header (4 bytes)
        let sub_hdr = SubmessageHeader {
            submsg_id: SubmessageId::WriteData,
            flags: 0x01, // Little endian
            length: (4 + topic_payload.len()) as u16,
        };
        sub_hdr.serialize(&mut out_buf[8..12])?;

        // 3. WriteData payload header: Request ID (0) + Object ID
        out_buf[12] = 0; // Request ID byte 0
        out_buf[13] = 0; // Request ID byte 1
        let obj_bytes = object_id.to_le_bytes();
        out_buf[14] = obj_bytes[0];
        out_buf[15] = obj_bytes[1];

        // 4. Topic payload bytes
        out_buf[16..total_len].copy_from_slice(topic_payload);

        Ok(total_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xrce_pack_and_unpack() {
        let hdr = MessageHeader {
            session_id: 0x81,
            stream_id: 0x01, // Best-effort
            sequence_nr: 42,
            client_key: [0x11, 0x22, 0x33, 0x44],
        };

        let payload = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut out = [0u8; 64];

        let written = WriteDataPayload::pack_frame(&hdr, 0x1234, &payload, &mut out).unwrap();
        assert_eq!(written, 8 + 4 + 4 + 4);

        let parsed_hdr = MessageHeader::deserialize(&out[..8]).unwrap();
        assert_eq!(parsed_hdr.session_id, 0x81);
        assert_eq!(parsed_hdr.sequence_nr, 42);
        assert_eq!(parsed_hdr.client_key, [0x11, 0x22, 0x33, 0x44]);

        let parsed_sub = SubmessageHeader::deserialize(&out[8..12]).unwrap();
        assert_eq!(parsed_sub.submsg_id, SubmessageId::WriteData);
        assert_eq!(parsed_sub.length, 8); // 4 (header) + 4 (data)
    }
}
