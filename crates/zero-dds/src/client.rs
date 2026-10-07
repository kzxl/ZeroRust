//! Micro-XRCE-DDS client session manager and publisher.

use crate::cdr::CdrWriter;
use crate::xrce::MessageHeader;
use zero_core::error::{ZeroError, ZeroResult};

/// Connection status of the Micro-ROS XRCE client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Not connected to Micro-XRCE Agent.
    Disconnected,
    /// Successfully handshaken and connected.
    Connected,
}

/// Micro-XRCE-DDS Client session for publishing ROS 2 topics over serial, UDP, or CAN.
#[derive(Debug)]
pub struct XrceClient {
    /// Unique 4-byte client key identifying this MCU node.
    pub client_key: [u8; 4],
    /// Session ID (typically 0x81).
    pub session_id: u8,
    /// Current session status.
    pub state: SessionState,
    /// Next sequence number for best-effort stream.
    best_effort_seq: u16,
}

impl XrceClient {
    /// Creates a new XRCE client with the given 4-byte client key.
    pub const fn new(client_key: [u8; 4]) -> Self {
        Self {
            client_key,
            session_id: 0x81,
            state: SessionState::Connected,
            best_effort_seq: 0,
        }
    }

    /// Allocates the next best-effort sequence number.
    pub fn next_sequence_nr(&mut self) -> u16 {
        let seq = self.best_effort_seq;
        self.best_effort_seq = self.best_effort_seq.wrapping_add(1);
        seq
    }

    /// Serializes and publishes a message payload to a DDS DataWriter entity.
    ///
    /// # Arguments
    /// - `writer_id`: Object ID of the DDS DataWriter configured in Micro-XRCE Agent.
    /// - `encode_msg`: Closure that writes the CDR message into a `CdrWriter`.
    /// - `out_buf`: Destination frame buffer.
    ///
    /// Returns total bytes written into `out_buf`.
    pub fn publish_message<F>(
        &mut self,
        writer_id: u16,
        encode_msg: F,
        out_buf: &mut [u8],
    ) -> ZeroResult<usize>
    where
        F: FnOnce(&mut CdrWriter) -> ZeroResult<()>,
    {
        if self.state != SessionState::Connected {
            return Err(ZeroError::HardwareFault);
        }

        // Submessage payload starts at offset 16 in out_buf
        if out_buf.len() < 24 {
            return Err(ZeroError::BufferOverflow);
        }

        // Encode CDR encapsulation header + message payload directly in-place
        let mut cdr = CdrWriter::new(&mut out_buf[16..]);
        cdr.write_header()?;
        encode_msg(&mut cdr)?;
        let topic_len = cdr.position();

        let header = MessageHeader {
            session_id: self.session_id,
            stream_id: 0x01, // Best-effort stream
            sequence_nr: self.next_sequence_nr(),
            client_key: self.client_key,
        };

        // Write header and submessage header around payload
        header.serialize(&mut out_buf[0..8])?;

        let sub_hdr = crate::xrce::SubmessageHeader {
            submsg_id: crate::xrce::SubmessageId::WriteData,
            flags: 0x01, // Little endian
            length: (4 + topic_len) as u16,
        };
        sub_hdr.serialize(&mut out_buf[8..12])?;

        // Request ID (0) + Object ID
        out_buf[12] = 0;
        out_buf[13] = 0;
        let obj_bytes = writer_id.to_le_bytes();
        out_buf[14] = obj_bytes[0];
        out_buf[15] = obj_bytes[1];

        Ok(16 + topic_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{Twist, Vector3};

    #[test]
    fn test_client_publish_twist() {
        let mut client = XrceClient::new([0x01, 0x02, 0x03, 0x04]);
        let mut frame_buf = [0u8; 128];

        let cmd_vel = Twist {
            linear: Vector3::new(1.0, 0.0, 0.0),
            angular: Vector3::new(0.0, 0.0, 0.5),
        };

        let written = client
            .publish_message(0x0100, |writer| cmd_vel.serialize(writer), &mut frame_buf)
            .unwrap();

        assert!(written > 16);
        assert_eq!(frame_buf[0], 0x81); // Session ID
        assert_eq!(frame_buf[1], 0x01); // Best effort stream
        assert_eq!(frame_buf[14], 0x00); // Writer ID low byte
        assert_eq!(frame_buf[15], 0x01); // Writer ID high byte
    }
}
