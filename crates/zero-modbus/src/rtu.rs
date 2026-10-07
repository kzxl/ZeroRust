//! Modbus RTU Frame encoding, decoding, and validation.

use crate::crc::calculate_crc16;
use crate::pdu::ModbusRequest;
use zero_core::error::{ZeroError, ZeroResult};

/// Modbus RTU Frame parser and serializer.
pub struct ModbusRtu;

impl ModbusRtu {
    /// Minimum valid Modbus RTU frame size (Slave ID + Function + 2 bytes data + 2 bytes CRC = 6 bytes).
    pub const MIN_FRAME_SIZE: usize = 6;

    /// Validates and parses an incoming raw Modbus RTU frame.
    ///
    /// Returns `(slave_address, ModbusRequest)`.
    pub fn parse_frame(frame: &[u8]) -> ZeroResult<(u8, ModbusRequest)> {
        if frame.len() < Self::MIN_FRAME_SIZE {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        // Verify CRC16: last 2 bytes are little-endian CRC
        let data_len = frame.len() - 2;
        let expected_crc = u16::from_le_bytes([frame[data_len], frame[data_len + 1]]);
        let computed_crc = calculate_crc16(&frame[..data_len]);

        if expected_crc != computed_crc {
            return Err(ZeroError::HardwareFault); // CRC mismatch
        }

        let slave_address = frame[0];
        let pdu_slice = &frame[1..data_len];
        let request = ModbusRequest::parse_pdu(pdu_slice)?;

        Ok((slave_address, request))
    }

    /// Encodes a Modbus Read Holding Registers response frame into `buffer`.
    ///
    /// Returns the total bytes written into `buffer`.
    pub fn encode_read_holding_response(
        slave_address: u8,
        registers: &[u16],
        buffer: &mut [u8],
    ) -> ZeroResult<usize> {
        let byte_count = (registers.len() * 2) as u8;
        let total_size = 1 + 1 + 1 + (byte_count as usize) + 2;

        if buffer.len() < total_size {
            return Err(ZeroError::BufferOverflow);
        }

        buffer[0] = slave_address;
        buffer[1] = 0x03; // Function Code
        buffer[2] = byte_count;

        let mut offset = 3;
        for &reg in registers {
            let be_bytes = reg.to_be_bytes();
            buffer[offset] = be_bytes[0];
            buffer[offset + 1] = be_bytes[1];
            offset += 2;
        }

        let crc = calculate_crc16(&buffer[..offset]);
        let crc_bytes = crc.to_le_bytes();
        buffer[offset] = crc_bytes[0];
        buffer[offset + 1] = crc_bytes[1];

        Ok(total_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtu_parse_valid_frame() {
        // Slave 1, Read Holding Registers (0x03), Start Addr 0x0002, Count 0x0004
        let mut frame = [0x01, 0x03, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00];
        let crc = calculate_crc16(&frame[..6]);
        let crc_bytes = crc.to_le_bytes();
        frame[6] = crc_bytes[0];
        frame[7] = crc_bytes[1];

        let (slave, req) = ModbusRtu::parse_frame(&frame).unwrap();
        assert_eq!(slave, 1);
        assert_eq!(
            req,
            ModbusRequest::ReadHoldingRegisters {
                address: 2,
                quantity: 4
            }
        );
    }
}
