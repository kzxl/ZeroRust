//! Pure Rust zero-allocation OMG CDR (Common Data Representation) serializer and deserializer.

use zero_core::error::{ZeroError, ZeroResult};

/// Standard CDR Encapsulation scheme identifiers (OMG CDR v1.5 / ROS 2).
pub struct CdrScheme;

impl CdrScheme {
    /// CDR Big Endian (0x0000).
    pub const CDR_BE: [u8; 4] = [0x00, 0x00, 0x00, 0x00];
    /// CDR Little Endian (0x0001).
    pub const CDR_LE: [u8; 4] = [0x00, 0x01, 0x00, 0x00];
}

/// Zero-allocation stream writer adhering to OMG CDR natural alignment rules.
pub struct CdrWriter<'a> {
    buffer: &'a mut [u8],
    cursor: usize,
}

impl<'a> CdrWriter<'a> {
    /// Creates a new CDR writer over the given mutable byte slice.
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self { buffer, cursor: 0 }
    }

    /// Number of bytes currently written into the buffer.
    pub const fn position(&self) -> usize {
        self.cursor
    }

    /// Aligns the cursor to a multiple of `align` bytes by writing padding zeros.
    pub fn align(&mut self, align: usize) -> ZeroResult<()> {
        let remainder = self.cursor % align;
        if remainder != 0 {
            let padding = align - remainder;
            if self.cursor + padding > self.buffer.len() {
                return Err(ZeroError::BufferOverflow);
            }
            self.buffer[self.cursor..self.cursor + padding].fill(0);
            self.cursor += padding;
        }
        Ok(())
    }

    /// Writes the 4-byte standard CDR encapsulation header (defaulting to Little Endian).
    pub fn write_header(&mut self) -> ZeroResult<()> {
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        self.buffer[self.cursor..self.cursor + 4].copy_from_slice(&CdrScheme::CDR_LE);
        self.cursor += 4;
        Ok(())
    }

    /// Writes a single boolean value (1 byte).
    pub fn write_bool(&mut self, val: bool) -> ZeroResult<()> {
        self.write_u8(if val { 1 } else { 0 })
    }

    /// Writes a single unsigned byte.
    pub fn write_u8(&mut self, val: u8) -> ZeroResult<()> {
        if self.cursor + 1 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        self.buffer[self.cursor] = val;
        self.cursor += 1;
        Ok(())
    }

    /// Writes a 16-bit integer with 2-byte alignment.
    pub fn write_u16(&mut self, val: u16) -> ZeroResult<()> {
        self.align(2)?;
        if self.cursor + 2 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        let bytes = val.to_le_bytes();
        self.buffer[self.cursor..self.cursor + 2].copy_from_slice(&bytes);
        self.cursor += 2;
        Ok(())
    }

    /// Writes a 32-bit signed integer with 4-byte alignment.
    pub fn write_i32(&mut self, val: i32) -> ZeroResult<()> {
        self.align(4)?;
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        let bytes = val.to_le_bytes();
        self.buffer[self.cursor..self.cursor + 4].copy_from_slice(&bytes);
        self.cursor += 4;
        Ok(())
    }

    /// Writes a 32-bit unsigned integer with 4-byte alignment.
    pub fn write_u32(&mut self, val: u32) -> ZeroResult<()> {
        self.align(4)?;
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        let bytes = val.to_le_bytes();
        self.buffer[self.cursor..self.cursor + 4].copy_from_slice(&bytes);
        self.cursor += 4;
        Ok(())
    }

    /// Writes a 32-bit IEEE float with 4-byte alignment.
    pub fn write_f32(&mut self, val: f32) -> ZeroResult<()> {
        self.align(4)?;
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        let bytes = val.to_le_bytes();
        self.buffer[self.cursor..self.cursor + 4].copy_from_slice(&bytes);
        self.cursor += 4;
        Ok(())
    }

    /// Writes a 64-bit IEEE float with 8-byte alignment.
    pub fn write_f64(&mut self, val: f64) -> ZeroResult<()> {
        self.align(8)?;
        if self.cursor + 8 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        let bytes = val.to_le_bytes();
        self.buffer[self.cursor..self.cursor + 8].copy_from_slice(&bytes);
        self.cursor += 8;
        Ok(())
    }

    /// Writes a CDR string: 4-byte length prefix (including trailing null byte) followed by UTF-8 bytes and `\0`.
    pub fn write_str(&mut self, s: &str) -> ZeroResult<()> {
        let bytes = s.as_bytes();
        let len_with_null = (bytes.len() + 1) as u32;
        self.write_u32(len_with_null)?;

        if self.cursor + bytes.len() + 1 > self.buffer.len() {
            return Err(ZeroError::BufferOverflow);
        }
        self.buffer[self.cursor..self.cursor + bytes.len()].copy_from_slice(bytes);
        self.cursor += bytes.len();
        self.buffer[self.cursor] = 0; // Null terminator
        self.cursor += 1;
        Ok(())
    }
}

/// Zero-allocation stream reader adhering to OMG CDR natural alignment rules.
pub struct CdrReader<'a> {
    buffer: &'a [u8],
    cursor: usize,
}

impl<'a> CdrReader<'a> {
    /// Creates a new CDR reader over the given byte slice.
    pub fn new(buffer: &'a [u8]) -> Self {
        Self { buffer, cursor: 0 }
    }

    /// Current cursor position in buffer.
    pub const fn position(&self) -> usize {
        self.cursor
    }

    /// Advances the cursor to the required alignment boundary.
    pub fn align(&mut self, align: usize) -> ZeroResult<()> {
        let remainder = self.cursor % align;
        if remainder != 0 {
            let padding = align - remainder;
            if self.cursor + padding > self.buffer.len() {
                return Err(ZeroError::UnexpectedEndOfBuffer);
            }
            self.cursor += padding;
        }
        Ok(())
    }

    /// Reads and verifies the 4-byte encapsulation header. Returns true for Little Endian.
    pub fn read_header(&mut self) -> ZeroResult<bool> {
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let is_le = self.buffer[self.cursor + 1] == 0x01;
        self.cursor += 4;
        Ok(is_le)
    }

    /// Reads a single boolean value.
    pub fn read_bool(&mut self) -> ZeroResult<bool> {
        let byte = self.read_u8()?;
        Ok(byte != 0)
    }

    /// Reads a single unsigned byte.
    pub fn read_u8(&mut self) -> ZeroResult<u8> {
        if self.cursor + 1 > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let b = self.buffer[self.cursor];
        self.cursor += 1;
        Ok(b)
    }

    /// Reads a 16-bit integer with 2-byte alignment.
    pub fn read_u16(&mut self) -> ZeroResult<u16> {
        self.align(2)?;
        if self.cursor + 2 > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let val = u16::from_le_bytes([self.buffer[self.cursor], self.buffer[self.cursor + 1]]);
        self.cursor += 2;
        Ok(val)
    }

    /// Reads a 32-bit signed integer with 4-byte alignment.
    pub fn read_i32(&mut self) -> ZeroResult<i32> {
        self.align(4)?;
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let val = i32::from_le_bytes([
            self.buffer[self.cursor],
            self.buffer[self.cursor + 1],
            self.buffer[self.cursor + 2],
            self.buffer[self.cursor + 3],
        ]);
        self.cursor += 4;
        Ok(val)
    }

    /// Reads a 32-bit unsigned integer with 4-byte alignment.
    pub fn read_u32(&mut self) -> ZeroResult<u32> {
        self.align(4)?;
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let val = u32::from_le_bytes([
            self.buffer[self.cursor],
            self.buffer[self.cursor + 1],
            self.buffer[self.cursor + 2],
            self.buffer[self.cursor + 3],
        ]);
        self.cursor += 4;
        Ok(val)
    }

    /// Reads a 32-bit IEEE float with 4-byte alignment.
    pub fn read_f32(&mut self) -> ZeroResult<f32> {
        self.align(4)?;
        if self.cursor + 4 > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let val = f32::from_le_bytes([
            self.buffer[self.cursor],
            self.buffer[self.cursor + 1],
            self.buffer[self.cursor + 2],
            self.buffer[self.cursor + 3],
        ]);
        self.cursor += 4;
        Ok(val)
    }

    /// Reads a 64-bit IEEE float with 8-byte alignment.
    pub fn read_f64(&mut self) -> ZeroResult<f64> {
        self.align(8)?;
        if self.cursor + 8 > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let val = f64::from_le_bytes([
            self.buffer[self.cursor],
            self.buffer[self.cursor + 1],
            self.buffer[self.cursor + 2],
            self.buffer[self.cursor + 3],
            self.buffer[self.cursor + 4],
            self.buffer[self.cursor + 5],
            self.buffer[self.cursor + 6],
            self.buffer[self.cursor + 7],
        ]);
        self.cursor += 8;
        Ok(val)
    }

    /// Reads a CDR string slice without allocating heap memory.
    pub fn read_str(&mut self) -> ZeroResult<&'a str> {
        let len_with_null = self.read_u32()? as usize;
        if len_with_null == 0 {
            return Ok("");
        }
        let str_len = len_with_null - 1; // Exclude trailing \0
        if self.cursor + len_with_null > self.buffer.len() {
            return Err(ZeroError::UnexpectedEndOfBuffer);
        }
        let str_bytes = &self.buffer[self.cursor..self.cursor + str_len];
        let s = core::str::from_utf8(str_bytes).map_err(|_| ZeroError::InvalidArgument)?;
        self.cursor += len_with_null;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdr_alignment_and_primitives() {
        let mut buf = [0u8; 64];
        let mut writer = CdrWriter::new(&mut buf);

        writer.write_header().unwrap();
        writer.write_u8(0x42).unwrap(); // Offset 4 + 1 = 5
        writer.write_u32(0x12345678).unwrap(); // Aligns to 8, writes 4 -> Offset 12
        writer.write_f64(core::f64::consts::PI).unwrap(); // Aligns to 16, writes 8 -> Offset 24
        writer.write_str("ZeroRust").unwrap();

        let total_len = writer.position();

        let mut reader = CdrReader::new(&buf[..total_len]);
        assert!(reader.read_header().unwrap());
        assert_eq!(reader.read_u8().unwrap(), 0x42);
        assert_eq!(reader.read_u32().unwrap(), 0x12345678);
        assert!((reader.read_f64().unwrap() - core::f64::consts::PI).abs() < 1e-9);
        assert_eq!(reader.read_str().unwrap(), "ZeroRust");
    }
}
