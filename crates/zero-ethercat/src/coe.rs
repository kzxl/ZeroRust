//! CANopen over EtherCAT (CoE) mailbox protocol framing and SDO transactions.

use zero_core::error::{ZeroError, ZeroResult};

/// Mailbox protocol types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MailboxType {
    /// Error response (0x00).
    Error = 0x00,
    /// Ethernet over EtherCAT (0x01).
    EoE = 0x01,
    /// CANopen over EtherCAT (0x03).
    CoE = 0x03,
    /// File access over EtherCAT (0x04).
    FoE = 0x04,
    /// Servo drive profile over EtherCAT (0x05).
    SoE = 0x05,
    /// Vendor specific (0x0F).
    VoE = 0x0F,
}

/// CoE Service types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CoeService {
    /// Emergency message (0x01).
    Emergency = 0x01,
    /// SDO Request from master to slave (0x02).
    SdoRequest = 0x02,
    /// SDO Response from slave to master (0x03).
    SdoResponse = 0x03,
    /// Transmit PDO (0x04).
    TxPdo = 0x04,
    /// Receive PDO (0x05).
    RxPdo = 0x05,
    /// SDO Information service (0x08).
    SdoInformation = 0x08,
}

impl TryFrom<u8> for CoeService {
    type Error = ZeroError;

    fn try_from(val: u8) -> Result<Self, Self::Error> {
        match val {
            0x01 => Ok(Self::Emergency),
            0x02 => Ok(Self::SdoRequest),
            0x03 => Ok(Self::SdoResponse),
            0x04 => Ok(Self::TxPdo),
            0x05 => Ok(Self::RxPdo),
            0x08 => Ok(Self::SdoInformation),
            _ => Err(ZeroError::InvalidArgument),
        }
    }
}

/// Standard 6-byte EtherCAT Mailbox Header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MailboxHeader {
    /// Length of mailbox service payload in bytes (excluding this 6-byte header).
    pub length: u16,
    /// Station address of recipient.
    pub address: u16,
    /// Channel and Priority (bits 0..5), Type (bits 6..9), Counter (bits 12..14).
    pub channel_prio: u8,
    /// Mailbox type (e.g. CoE).
    pub mbx_type: MailboxType,
    /// Rolling sequence counter (1..7).
    pub counter: u8,
}

impl MailboxHeader {
    /// Serializes mailbox header into a 6-byte buffer.
    pub fn serialize(&self, buf: &mut [u8]) -> ZeroResult<()> {
        if buf.len() < 6 {
            return Err(ZeroError::BufferOverflow);
        }
        let len_bytes = self.length.to_le_bytes();
        buf[0] = len_bytes[0];
        buf[1] = len_bytes[1];

        let addr_bytes = self.address.to_le_bytes();
        buf[2] = addr_bytes[0];
        buf[3] = addr_bytes[1];

        // Byte 4: bits 0..5 channel/prio, bits 6..7 low 2 bits of type
        let type_val = self.mbx_type as u8;
        buf[4] = (self.channel_prio & 0x3F) | ((type_val & 0x03) << 6);

        // Byte 5: bits 0..1 high 2 bits of type, bits 4..6 counter
        buf[5] = ((type_val >> 2) & 0x03) | ((self.counter & 0x07) << 4);

        Ok(())
    }

    /// Deserializes mailbox header from a 6-byte buffer.
    pub fn deserialize(buf: &[u8]) -> ZeroResult<Self> {
        if buf.len() < 6 {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let length = u16::from_le_bytes([buf[0], buf[1]]);
        let address = u16::from_le_bytes([buf[2], buf[3]]);
        let channel_prio = buf[4] & 0x3F;
        let type_low = (buf[4] >> 6) & 0x03;
        let type_high = buf[5] & 0x03;
        let mbx_type_raw = (type_high << 2) | type_low;
        let counter = (buf[5] >> 4) & 0x07;

        let mbx_type = match mbx_type_raw {
            0x00 => MailboxType::Error,
            0x01 => MailboxType::EoE,
            0x03 => MailboxType::CoE,
            0x04 => MailboxType::FoE,
            0x05 => MailboxType::SoE,
            0x0F => MailboxType::VoE,
            _ => return Err(ZeroError::InvalidArgument),
        };

        Ok(Self {
            length,
            address,
            channel_prio,
            mbx_type,
            counter,
        })
    }
}

/// SDO Download (Write) or Upload (Read) transaction builder for CoE.
pub struct CoeSdo;

impl CoeSdo {
    /// Builds an expedited SDO Download (write up to 4 bytes) request into a mailbox buffer.
    ///
    /// Total buffer size needed: 6 (Mailbox) + 2 (CoE Header) + 8 (SDO) = 16 bytes.
    pub fn build_download_request(
        dest_addr: u16,
        counter: u8,
        index: u16,
        subindex: u8,
        value: u32,
        size_bytes: u8,
        buf: &mut [u8],
    ) -> ZeroResult<usize> {
        if size_bytes == 0 || size_bytes > 4 {
            return Err(ZeroError::InvalidArgument);
        }
        if buf.len() < 16 {
            return Err(ZeroError::BufferOverflow);
        }

        let mbx = MailboxHeader {
            length: 10, // 2 (CoE Header) + 8 (SDO)
            address: dest_addr,
            channel_prio: 0,
            mbx_type: MailboxType::CoE,
            counter,
        };
        mbx.serialize(&mut buf[0..6])?;

        // CoE Header (2 bytes): Number (9 bits) + Service (4 bits at bit 12..15)
        let coe_header: u16 = (CoeService::SdoRequest as u16) << 12;
        let coe_bytes = coe_header.to_le_bytes();
        buf[6] = coe_bytes[0];
        buf[7] = coe_bytes[1];

        // SDO Expedited Download Command Specifier
        // CCS = 1 (bits 7-5 = 001) -> 0x20
        // e = 1 (bit 1 = 1) -> 0x02
        // s = 1 (bit 0 = 1) -> 0x01
        // n = 4 - size_bytes (bits 3-2)
        let n = 4 - size_bytes;
        let cs = 0x20 | (n << 2) | 0x02 | 0x01;
        buf[8] = cs;

        let idx_bytes = index.to_le_bytes();
        buf[9] = idx_bytes[0];
        buf[10] = idx_bytes[1];
        buf[11] = subindex;

        let val_bytes = value.to_le_bytes();
        buf[12..16].copy_from_slice(&val_bytes);

        Ok(16)
    }

    /// Builds an expedited SDO Upload (read) request into a mailbox buffer.
    pub fn build_upload_request(
        dest_addr: u16,
        counter: u8,
        index: u16,
        subindex: u8,
        buf: &mut [u8],
    ) -> ZeroResult<usize> {
        if buf.len() < 16 {
            return Err(ZeroError::BufferOverflow);
        }

        let mbx = MailboxHeader {
            length: 10,
            address: dest_addr,
            channel_prio: 0,
            mbx_type: MailboxType::CoE,
            counter,
        };
        mbx.serialize(&mut buf[0..6])?;

        let coe_header: u16 = (CoeService::SdoRequest as u16) << 12;
        let coe_bytes = coe_header.to_le_bytes();
        buf[6] = coe_bytes[0];
        buf[7] = coe_bytes[1];

        // SDO Upload Command Specifier: CCS = 2 -> 0x40
        buf[8] = 0x40;
        let idx_bytes = index.to_le_bytes();
        buf[9] = idx_bytes[0];
        buf[10] = idx_bytes[1];
        buf[11] = subindex;
        buf[12..16].fill(0);

        Ok(16)
    }

    /// Parses an SDO Upload response from a received mailbox buffer.
    ///
    /// Returns `(index, subindex, value, size_bytes)`.
    pub fn parse_upload_response(buf: &[u8]) -> ZeroResult<(u16, u8, u32, u8)> {
        if buf.len() < 16 {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        let mbx = MailboxHeader::deserialize(&buf[0..6])?;
        if mbx.mbx_type != MailboxType::CoE {
            return Err(ZeroError::InvalidArgument);
        }

        let coe_word = u16::from_le_bytes([buf[6], buf[7]]);
        let service = (coe_word >> 12) as u8;
        if service != CoeService::SdoResponse as u8 {
            return Err(ZeroError::InvalidArgument);
        }

        let cs = buf[8];
        // Check for SDO Abort (0x80)
        if cs == 0x80 {
            let abort_code = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
            let _ = abort_code;
            return Err(ZeroError::HardwareFault);
        }

        // SCS = 2 (bits 7-5 = 010 -> 0x40)
        if (cs & 0xE0) != 0x40 {
            return Err(ZeroError::InvalidArgument);
        }

        let index = u16::from_le_bytes([buf[9], buf[10]]);
        let subindex = buf[11];

        let is_expedited = (cs & 0x02) != 0;
        let size_indicated = (cs & 0x01) != 0;

        let size_bytes = if is_expedited && size_indicated {
            let n = (cs >> 2) & 0x03;
            4 - n
        } else {
            4
        };

        let value = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

        Ok((index, subindex, value, size_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mailbox_header_roundtrip() {
        let mbx = MailboxHeader {
            length: 10,
            address: 0x1001,
            channel_prio: 0x00,
            mbx_type: MailboxType::CoE,
            counter: 3,
        };

        let mut buf = [0u8; 6];
        mbx.serialize(&mut buf).unwrap();

        let parsed = MailboxHeader::deserialize(&buf).unwrap();
        assert_eq!(parsed.length, 10);
        assert_eq!(parsed.address, 0x1001);
        assert_eq!(parsed.mbx_type, MailboxType::CoE);
        assert_eq!(parsed.counter, 3);
    }

    #[test]
    fn test_coe_sdo_download_and_upload() {
        let mut buf = [0u8; 32];
        let written = CoeSdo::build_download_request(
            0x1001, 1, 0x6040, // CiA 402 Controlword
            0x00, 0x000F, // Enable Operation
            2, &mut buf,
        )
        .unwrap();

        assert_eq!(written, 16);
        assert_eq!(buf[8], 0x2B); // SDO Expedited Download 2 bytes
        assert_eq!(buf[9], 0x40);
        assert_eq!(buf[10], 0x60);
        assert_eq!(buf[11], 0x00);
        assert_eq!(buf[12], 0x0F);
        assert_eq!(buf[13], 0x00);
    }
}
