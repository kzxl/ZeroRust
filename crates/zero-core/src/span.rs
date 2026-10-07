//! Zero-copy byte buffer reading and writing utilities for binary protocol frames.

use crate::error::{ZeroError, ZeroResult};

/// Cursor-based zero-copy reader over a byte slice.
pub struct ByteReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    /// Creates a new `ByteReader` wrapping the given byte slice.
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    /// Returns the remaining unread bytes.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    /// Returns the current offset cursor.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Reads a single `u8`.
    pub fn read_u8(&mut self) -> ZeroResult<u8> {
        if self.offset >= self.data.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let val = self.data[self.offset];
        self.offset += 1;
        Ok(val)
    }

    /// Reads a `u16` in little-endian format.
    pub fn read_u16_le(&mut self) -> ZeroResult<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Reads a `u16` in big-endian format.
    pub fn read_u16_be(&mut self) -> ZeroResult<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    /// Reads a `u32` in little-endian format.
    pub fn read_u32_le(&mut self) -> ZeroResult<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Reads a `u32` in big-endian format.
    pub fn read_u32_be(&mut self) -> ZeroResult<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Reads an `i32` in little-endian format.
    pub fn read_i32_le(&mut self) -> ZeroResult<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Reads an `i32` in big-endian format.
    pub fn read_i32_be(&mut self) -> ZeroResult<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Reads an `f32` in little-endian format.
    pub fn read_f32_le(&mut self) -> ZeroResult<f32> {
        let bits = self.read_u32_le()?;
        Ok(f32::from_bits(bits))
    }

    /// Reads an exact number of bytes as a subslice.
    pub fn read_bytes(&mut self, len: usize) -> ZeroResult<&'a [u8]> {
        if self.offset + len > self.data.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let slice = &self.data[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }
}

/// Cursor-based zero-copy writer into a mutable byte buffer.
pub struct ByteWriter<'a> {
    data: &'a mut [u8],
    offset: usize,
}

impl<'a> ByteWriter<'a> {
    /// Creates a new `ByteWriter` wrapping the given mutable slice.
    #[inline]
    pub fn new(data: &'a mut [u8]) -> Self {
        Self { data, offset: 0 }
    }

    /// Returns the number of bytes written so far.
    #[inline]
    pub fn bytes_written(&self) -> usize {
        self.offset
    }

    /// Returns remaining writable capacity.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    /// Writes a single `u8`.
    pub fn write_u8(&mut self, val: u8) -> ZeroResult<()> {
        if self.offset >= self.data.len() {
            return Err(ZeroError::BufferOverflow);
        }
        self.data[self.offset] = val;
        self.offset += 1;
        Ok(())
    }

    /// Writes a `u16` in little-endian format.
    pub fn write_u16_le(&mut self, val: u16) -> ZeroResult<()> {
        self.write_bytes(&val.to_le_bytes())
    }

    /// Writes a `u16` in big-endian format.
    pub fn write_u16_be(&mut self, val: u16) -> ZeroResult<()> {
        self.write_bytes(&val.to_be_bytes())
    }

    /// Writes a `u32` in little-endian format.
    pub fn write_u32_le(&mut self, val: u32) -> ZeroResult<()> {
        self.write_bytes(&val.to_le_bytes())
    }

    /// Writes a `u32` in big-endian format.
    pub fn write_u32_be(&mut self, val: u32) -> ZeroResult<()> {
        self.write_bytes(&val.to_be_bytes())
    }

    /// Writes an `i32` in little-endian format.
    pub fn write_i32_le(&mut self, val: i32) -> ZeroResult<()> {
        self.write_bytes(&val.to_le_bytes())
    }

    /// Writes an `i32` in big-endian format.
    pub fn write_i32_be(&mut self, val: i32) -> ZeroResult<()> {
        self.write_bytes(&val.to_be_bytes())
    }

    /// Writes an `f32` in little-endian format.
    pub fn write_f32_le(&mut self, val: f32) -> ZeroResult<()> {
        self.write_bytes(&val.to_bits().to_le_bytes())
    }

    /// Writes a sequence of bytes into the buffer.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> ZeroResult<()> {
        if self.offset + bytes.len() > self.data.len() {
            return Err(ZeroError::BufferOverflow);
        }
        self.data[self.offset..self.offset + bytes.len()].copy_from_slice(bytes);
        self.offset += bytes.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_reader_writer_roundtrip() {
        let mut buf = [0u8; 32];
        {
            let mut writer = ByteWriter::new(&mut buf);
            writer.write_u8(0xAA).unwrap();
            writer.write_u16_le(0x1234).unwrap();
            writer.write_u32_be(0xDEADBEEF).unwrap();
            writer.write_i32_le(-123456).unwrap();
            assert_eq!(writer.bytes_written(), 1 + 2 + 4 + 4);
        }

        let mut reader = ByteReader::new(&buf);
        assert_eq!(reader.read_u8().unwrap(), 0xAA);
        assert_eq!(reader.read_u16_le().unwrap(), 0x1234);
        assert_eq!(reader.read_u32_be().unwrap(), 0xDEADBEEF);
        assert_eq!(reader.read_i32_le().unwrap(), -123456);
    }
}
