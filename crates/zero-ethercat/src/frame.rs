//! EtherCAT IEEE 802.3 Frame and Datagram parsing and serialization.

use zero_core::error::{ZeroError, ZeroResult};

/// Standard EtherType for EtherCAT frames (0x88A4).
pub const ETHERTYPE_ETHERCAT: u16 = 0x88A4;

/// EtherCAT Datagram Command Codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EthercatCmd {
    /// Auto Increment Read (APRD = 0x01).
    Aprd = 0x01,
    /// Auto Increment Write (APWR = 0x02).
    Apwr = 0x02,
    /// Auto Increment Read Write (APRW = 0x03).
    Aprw = 0x03,
    /// Configured Address Read (FPRD = 0x04).
    Fprd = 0x04,
    /// Configured Address Write (FPWR = 0x05).
    Fpwr = 0x05,
    /// Configured Address Read Write (FPRW = 0x06).
    Fprw = 0x06,
    /// Broadcast Read (BRD = 0x07).
    Brd = 0x07,
    /// Broadcast Write (BWR = 0x08).
    Bwr = 0x08,
    /// Broadcast Read Write (BRW = 0x09).
    Brw = 0x09,
    /// Logical Memory Read (LRD = 0x0A).
    Lrd = 0x0A,
    /// Logical Memory Write (LWR = 0x0B).
    Lwr = 0x0B,
    /// Logical Memory Read Write (LRW = 0x0C).
    Lrw = 0x0C,
    /// Auto Increment Read Multiple Write (ARMW = 0x0D).
    Armw = 0x0D,
}

impl TryFrom<u8> for EthercatCmd {
    type Error = ZeroError;

    fn try_from(val: u8) -> Result<Self, Self::Error> {
        match val {
            0x01 => Ok(Self::Aprd),
            0x02 => Ok(Self::Apwr),
            0x03 => Ok(Self::Aprw),
            0x04 => Ok(Self::Fprd),
            0x05 => Ok(Self::Fpwr),
            0x06 => Ok(Self::Fprw),
            0x07 => Ok(Self::Brd),
            0x08 => Ok(Self::Bwr),
            0x09 => Ok(Self::Brw),
            0x0A => Ok(Self::Lrd),
            0x0B => Ok(Self::Lwr),
            0x0C => Ok(Self::Lrw),
            0x0D => Ok(Self::Armw),
            _ => Err(ZeroError::InvalidArgument),
        }
    }
}

/// An individual EtherCAT Datagram within an EtherCAT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EthercatDatagram<'a> {
    /// Command code.
    pub cmd: EthercatCmd,
    /// Sequence index to correlate request with response.
    pub index: u8,
    /// Address field: Auto-increment or Configured Slave Address + Register Offset.
    pub address: u32,
    /// Payload data slice.
    pub data: &'a [u8],
    /// Working Counter (WKC) returned by slave devices.
    pub wkc: u16,
}

impl<'a> EthercatDatagram<'a> {
    /// Minimum header size of an individual datagram (Cmd + Idx + Addr + Len/Flags + IRQ = 10 bytes).
    pub const HEADER_SIZE: usize = 10;
    /// Footer size (WKC = 2 bytes).
    pub const FOOTER_SIZE: usize = 2;

    /// Serializes this datagram into `buffer`.
    ///
    /// `more_datagrams`: true if additional datagrams follow in the same frame.
    /// Returns total bytes written.
    pub fn serialize(&self, buffer: &mut [u8], more_datagrams: bool) -> ZeroResult<usize> {
        let total_size = Self::HEADER_SIZE + self.data.len() + Self::FOOTER_SIZE;
        if buffer.len() < total_size {
            return Err(ZeroError::BufferOverflow);
        }

        buffer[0] = self.cmd as u8;
        buffer[1] = self.index;

        let addr_bytes = self.address.to_le_bytes();
        buffer[2..6].copy_from_slice(&addr_bytes);

        // Length (11 bits), R (1 bit), C (1 bit), M (more flag - bit 15)
        let len_flags =
            (self.data.len() as u16 & 0x07FF) | if more_datagrams { 0x8000 } else { 0x0000 };
        let len_bytes = len_flags.to_le_bytes();
        buffer[6] = len_bytes[0];
        buffer[7] = len_bytes[1];

        // Interrupt register (2 bytes = 0)
        buffer[8] = 0;
        buffer[9] = 0;

        // Data payload
        buffer[10..10 + self.data.len()].copy_from_slice(self.data);

        // Working Counter (WKC)
        let wkc_offset = 10 + self.data.len();
        let wkc_bytes = self.wkc.to_le_bytes();
        buffer[wkc_offset] = wkc_bytes[0];
        buffer[wkc_offset + 1] = wkc_bytes[1];

        Ok(total_size)
    }

    /// Deserializes a datagram from a raw buffer slice.
    ///
    /// Returns `(EthercatDatagram, has_more_flag, total_consumed_bytes)`.
    pub fn deserialize(buffer: &'a [u8]) -> ZeroResult<(Self, bool, usize)> {
        if buffer.len() < Self::HEADER_SIZE + Self::FOOTER_SIZE {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        let cmd = EthercatCmd::try_from(buffer[0])?;
        let index = buffer[1];
        let address = u32::from_le_bytes([buffer[2], buffer[3], buffer[4], buffer[5]]);

        let len_flags = u16::from_le_bytes([buffer[6], buffer[7]]);
        let data_len = (len_flags & 0x07FF) as usize;
        let has_more = (len_flags & 0x8000) != 0;

        let total_size = Self::HEADER_SIZE + data_len + Self::FOOTER_SIZE;
        if buffer.len() < total_size {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }

        let data = &buffer[10..10 + data_len];
        let wkc_offset = 10 + data_len;
        let wkc = u16::from_le_bytes([buffer[wkc_offset], buffer[wkc_offset + 1]]);

        Ok((
            Self {
                cmd,
                index,
                address,
                data,
                wkc,
            },
            has_more,
            total_size,
        ))
    }
}

/// Helper to wrap one or more datagrams inside an EtherCAT Ethernet payload (Header + Datagrams).
pub struct EthercatFrame;

impl EthercatFrame {
    /// EtherCAT Frame Header size (2 bytes: Length 11 bits + Type 4 bits).
    pub const FRAME_HEADER_SIZE: usize = 2;

    /// Encodes an EtherCAT Frame header into `buffer`.
    pub fn encode_frame_header(datagram_total_len: u16, buffer: &mut [u8]) -> ZeroResult<()> {
        if buffer.len() < Self::FRAME_HEADER_SIZE {
            return Err(ZeroError::BufferOverflow);
        }
        // Type 1 = EtherCAT commands (bits 12..15 = 0x1)
        let header = (datagram_total_len & 0x07FF) | (0x1000);
        let bytes = header.to_le_bytes();
        buffer[0] = bytes[0];
        buffer[1] = bytes[1];
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datagram_roundtrip() {
        let payload = [0x11, 0x22, 0x33, 0x44];
        let dgram = EthercatDatagram {
            cmd: EthercatCmd::Brd,
            index: 0x42,
            address: 0x0000_0130, // AL Status Register
            data: &payload,
            wkc: 1,
        };

        let mut buf = [0u8; 64];
        let written = dgram.serialize(&mut buf, false).unwrap();
        assert_eq!(written, 10 + 4 + 2);

        let (parsed, has_more, consumed) = EthercatDatagram::deserialize(&buf[..written]).unwrap();
        assert_eq!(consumed, written);
        assert!(!has_more);
        assert_eq!(parsed.cmd, EthercatCmd::Brd);
        assert_eq!(parsed.index, 0x42);
        assert_eq!(parsed.address, 0x0000_0130);
        assert_eq!(parsed.data, &payload);
        assert_eq!(parsed.wkc, 1);
    }
}
