//! FastCGI v1.0 binary protocol records, headers, and name-value pair encoding.

use zero_core::error::{ZeroError, ZeroResult};

/// Standard FastCGI protocol version (always 1).
pub const FCGI_VERSION_1: u8 = 1;

/// FastCGI record types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FcgiRecordType {
    /// Begins a new request (1).
    BeginRequest = 1,
    /// Aborts an in-flight request (2).
    AbortRequest = 2,
    /// Signals normal or abnormal request completion (3).
    EndRequest = 3,
    /// Streams environment / CGI parameter pairs (4).
    Params = 4,
    /// Streams HTTP request body (5).
    Stdin = 5,
    /// Streams HTTP response body from PHP-FPM (6).
    Stdout = 6,
    /// Streams error logging output from PHP-FPM (7).
    Stderr = 7,
    /// Additional application data (8).
    Data = 8,
    /// Query server capabilities (9).
    GetValues = 9,
    /// Response to capabilities query (10).
    GetValuesResult = 10,
    /// Unknown record type notification (11).
    UnknownType = 11,
}

impl TryFrom<u8> for FcgiRecordType {
    type Error = ZeroError;

    fn try_from(val: u8) -> Result<Self, Self::Error> {
        match val {
            1 => Ok(Self::BeginRequest),
            2 => Ok(Self::AbortRequest),
            3 => Ok(Self::EndRequest),
            4 => Ok(Self::Params),
            5 => Ok(Self::Stdin),
            6 => Ok(Self::Stdout),
            7 => Ok(Self::Stderr),
            8 => Ok(Self::Data),
            9 => Ok(Self::GetValues),
            10 => Ok(Self::GetValuesResult),
            11 => Ok(Self::UnknownType),
            _ => Err(ZeroError::InvalidArgument),
        }
    }
}

/// Standard 8-byte FastCGI record header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FcgiHeader {
    /// Protocol version (FCGI_VERSION_1 = 1).
    pub version: u8,
    /// Record type.
    pub record_type: FcgiRecordType,
    /// Request identifier correlating stream chunks.
    pub request_id: u16,
    /// Payload length in bytes (excluding 8-byte header and padding).
    pub content_length: u16,
    /// Number of trailing padding bytes (0..255).
    pub padding_length: u8,
}

impl FcgiHeader {
    /// Header size in bytes.
    pub const SIZE: usize = 8;

    /// Serializes 8-byte FastCGI header into buffer.
    pub fn serialize(&self, buf: &mut [u8]) -> ZeroResult<()> {
        if buf.len() < Self::SIZE {
            return Err(ZeroError::BufferOverflow);
        }
        buf[0] = self.version;
        buf[1] = self.record_type as u8;
        let req_bytes = self.request_id.to_be_bytes();
        buf[2] = req_bytes[0];
        buf[3] = req_bytes[1];
        let len_bytes = self.content_length.to_be_bytes();
        buf[4] = len_bytes[0];
        buf[5] = len_bytes[1];
        buf[6] = self.padding_length;
        buf[7] = 0; // Reserved
        Ok(())
    }

    /// Deserializes 8-byte FastCGI header from buffer.
    pub fn deserialize(buf: &[u8]) -> ZeroResult<Self> {
        if buf.len() < Self::SIZE {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let version = buf[0];
        let record_type = FcgiRecordType::try_from(buf[1])?;
        let request_id = u16::from_be_bytes([buf[2], buf[3]]);
        let content_length = u16::from_be_bytes([buf[4], buf[5]]);
        let padding_length = buf[6];
        Ok(Self {
            version,
            record_type,
            request_id,
            content_length,
            padding_length,
        })
    }
}

/// FastCGI Name-Value pair serialization helper.
pub struct FcgiNameValuePair;

impl FcgiNameValuePair {
    /// Encodes a name-value pair into `buf`.
    ///
    /// Encodes length as 1 byte if `< 128`, or 4 bytes with top bit set if `>= 128`.
    pub fn encode(name: &[u8], value: &[u8], buf: &mut [u8]) -> ZeroResult<usize> {
        let name_len_hdr = if name.len() < 128 { 1 } else { 4 };
        let val_len_hdr = if value.len() < 128 { 1 } else { 4 };
        let total_size = name_len_hdr + val_len_hdr + name.len() + value.len();

        if buf.len() < total_size {
            return Err(ZeroError::BufferOverflow);
        }

        let mut offset = 0;

        // Write name length
        if name.len() < 128 {
            buf[offset] = name.len() as u8;
            offset += 1;
        } else {
            let encoded = (name.len() as u32) | 0x8000_0000;
            buf[offset..offset + 4].copy_from_slice(&encoded.to_be_bytes());
            offset += 4;
        }

        // Write value length
        if value.len() < 128 {
            buf[offset] = value.len() as u8;
            offset += 1;
        } else {
            let encoded = (value.len() as u32) | 0x8000_0000;
            buf[offset..offset + 4].copy_from_slice(&encoded.to_be_bytes());
            offset += 4;
        }

        // Copy name bytes
        buf[offset..offset + name.len()].copy_from_slice(name);
        offset += name.len();

        // Copy value bytes
        buf[offset..offset + value.len()].copy_from_slice(value);
        offset += value.len();

        Ok(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = FcgiHeader {
            version: FCGI_VERSION_1,
            record_type: FcgiRecordType::BeginRequest,
            request_id: 0x0102,
            content_length: 8,
            padding_length: 0,
        };

        let mut buf = [0u8; 8];
        header.serialize(&mut buf).unwrap();

        let parsed = FcgiHeader::deserialize(&buf).unwrap();
        assert_eq!(parsed.version, FCGI_VERSION_1);
        assert_eq!(parsed.record_type, FcgiRecordType::BeginRequest);
        assert_eq!(parsed.request_id, 0x0102);
        assert_eq!(parsed.content_length, 8);
    }

    #[test]
    fn test_name_value_encode() {
        let mut buf = [0u8; 64];
        let written =
            FcgiNameValuePair::encode(b"SCRIPT_FILENAME", b"/var/www/index.php", &mut buf).unwrap();

        assert_eq!(written, 1 + 1 + 15 + 18);
        assert_eq!(buf[0], 15); // name len < 128
        assert_eq!(buf[1], 18); // val len < 128
        assert_eq!(&buf[2..17], b"SCRIPT_FILENAME");
        assert_eq!(&buf[17..35], b"/var/www/index.php");
    }
}
