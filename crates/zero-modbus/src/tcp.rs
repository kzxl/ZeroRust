//! Modbus TCP (MBAP) Header encoding and decoding.

use crate::pdu::ModbusRequest;
use zero_core::error::{ZeroError, ZeroResult};

/// Modbus Application Protocol (MBAP) Header for Modbus TCP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbapHeader {
    /// Transaction Identifier used for transaction pairing.
    pub transaction_id: u16,
    /// Protocol Identifier: 0 = Modbus protocol.
    pub protocol_id: u16,
    /// Length of remainder of frame (Unit ID + PDU).
    pub length: u16,
    /// Unit Identifier / Slave address for routing.
    pub unit_id: u8,
}

impl MbapHeader {
    /// Standard MBAP Header size in bytes.
    pub const HEADER_SIZE: usize = 7;

    /// Parses an MBAP header from raw bytes.
    pub fn parse(bytes: &[u8]) -> ZeroResult<Self> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        let transaction_id = u16::from_be_bytes([bytes[0], bytes[1]]);
        let protocol_id = u16::from_be_bytes([bytes[2], bytes[3]]);
        let length = u16::from_be_bytes([bytes[4], bytes[5]]);
        let unit_id = bytes[6];

        if protocol_id != 0 {
            return Err(ZeroError::Unsupported);
        }

        Ok(Self {
            transaction_id,
            protocol_id,
            length,
            unit_id,
        })
    }

    /// Serializes the MBAP header into `buffer`.
    pub fn encode(&self, buffer: &mut [u8]) -> ZeroResult<()> {
        if buffer.len() < Self::HEADER_SIZE {
            return Err(ZeroError::BufferOverflow);
        }

        let tid = self.transaction_id.to_be_bytes();
        let pid = self.protocol_id.to_be_bytes();
        let len = self.length.to_be_bytes();

        buffer[0] = tid[0];
        buffer[1] = tid[1];
        buffer[2] = pid[0];
        buffer[3] = pid[1];
        buffer[4] = len[0];
        buffer[5] = len[1];
        buffer[6] = self.unit_id;

        Ok(())
    }
}

/// Modbus TCP request wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModbusTcpFrame {
    /// Header information.
    pub header: MbapHeader,
    /// Parsed request body.
    pub request: ModbusRequest,
}

impl ModbusTcpFrame {
    /// Parses a complete Modbus TCP packet (MBAP + PDU).
    pub fn parse(packet: &[u8]) -> ZeroResult<Self> {
        let header = MbapHeader::parse(packet)?;
        let pdu_start = MbapHeader::HEADER_SIZE;
        let pdu_end = pdu_start + (header.length as usize).saturating_sub(1);

        if packet.len() < pdu_end {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        let request = ModbusRequest::parse_pdu(&packet[pdu_start..pdu_end])?;
        Ok(Self { header, request })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbap_header_roundtrip() {
        let header = MbapHeader {
            transaction_id: 0x1020,
            protocol_id: 0,
            length: 6,
            unit_id: 1,
        };

        let mut buf = [0u8; 7];
        header.encode(&mut buf).unwrap();

        let parsed = MbapHeader::parse(&buf).unwrap();
        assert_eq!(parsed, header);
    }
}
