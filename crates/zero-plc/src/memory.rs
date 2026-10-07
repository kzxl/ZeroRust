//! IEC 61131-3 Process Image Memory Model (%I Inputs, %Q Outputs, %M Markers).

use zero_core::error::{ZeroError, ZeroResult};

/// IEC 61131-3 memory areas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlcArea {
    /// %I Physical Inputs process image.
    Inputs,
    /// %Q Physical Outputs process image.
    Outputs,
    /// %M Internal Memory / Markers (flags, state words).
    Markers,
}

/// Structured PLC variable address referencing bits, bytes, words, or floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlcAddress {
    /// Memory area (%I, %Q, %M).
    pub area: PlcArea,
    /// Byte offset within the area.
    pub byte_offset: usize,
    /// Bit offset (0..7) for boolean operations.
    pub bit_offset: u8,
}

impl PlcAddress {
    /// Creates a bit-level address (e.g. %IX0.2 or %QX1.5).
    pub const fn bit(area: PlcArea, byte_offset: usize, bit_offset: u8) -> Self {
        Self {
            area,
            byte_offset,
            bit_offset,
        }
    }

    /// Creates a byte-level address (e.g. %IB0, %QB1, %MB10).
    pub const fn byte(area: PlcArea, byte_offset: usize) -> Self {
        Self {
            area,
            byte_offset,
            bit_offset: 0,
        }
    }

    /// Creates a 16-bit word address (e.g. %IW0, %QW2).
    pub const fn word(area: PlcArea, byte_offset: usize) -> Self {
        Self {
            area,
            byte_offset,
            bit_offset: 0,
        }
    }

    /// Creates a 32-bit double-word or float address (e.g. %ID0, %QD4, %MD100).
    pub const fn dword(area: PlcArea, byte_offset: usize) -> Self {
        Self {
            area,
            byte_offset,
            bit_offset: 0,
        }
    }
}

/// Zero-allocation IEC 61131-3 process image memory store.
#[derive(Debug)]
pub struct PlcMemory<const IN_BYTES: usize, const OUT_BYTES: usize, const MARKER_BYTES: usize> {
    /// %I Process Input Image.
    pub inputs: [u8; IN_BYTES],
    /// %Q Process Output Image.
    pub outputs: [u8; OUT_BYTES],
    /// %M Internal Markers / Flags.
    pub markers: [u8; MARKER_BYTES],
}

impl<const IN_BYTES: usize, const OUT_BYTES: usize, const MARKER_BYTES: usize> Default
    for PlcMemory<IN_BYTES, OUT_BYTES, MARKER_BYTES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const IN_BYTES: usize, const OUT_BYTES: usize, const MARKER_BYTES: usize>
    PlcMemory<IN_BYTES, OUT_BYTES, MARKER_BYTES>
{
    /// Creates zero-initialized PLC memory images.
    pub const fn new() -> Self {
        Self {
            inputs: [0u8; IN_BYTES],
            outputs: [0u8; OUT_BYTES],
            markers: [0u8; MARKER_BYTES],
        }
    }

    fn slice_for_area(&self, area: PlcArea) -> &[u8] {
        match area {
            PlcArea::Inputs => &self.inputs,
            PlcArea::Outputs => &self.outputs,
            PlcArea::Markers => &self.markers,
        }
    }

    fn slice_for_area_mut(&mut self, area: PlcArea) -> &mut [u8] {
        match area {
            PlcArea::Inputs => &mut self.inputs,
            PlcArea::Outputs => &mut self.outputs,
            PlcArea::Markers => &mut self.markers,
        }
    }

    /// Reads an individual boolean bit (%IX, %QX, %MX).
    pub fn read_bit(&self, addr: PlcAddress) -> ZeroResult<bool> {
        let slice = self.slice_for_area(addr.area);
        if addr.byte_offset >= slice.len() || addr.bit_offset > 7 {
            return Err(ZeroError::InvalidArgument);
        }
        let byte = slice[addr.byte_offset];
        Ok((byte & (1 << addr.bit_offset)) != 0)
    }

    /// Writes an individual boolean bit (%IX, %QX, %MX).
    pub fn write_bit(&mut self, addr: PlcAddress, value: bool) -> ZeroResult<()> {
        let slice = self.slice_for_area_mut(addr.area);
        if addr.byte_offset >= slice.len() || addr.bit_offset > 7 {
            return Err(ZeroError::InvalidArgument);
        }
        if value {
            slice[addr.byte_offset] |= 1 << addr.bit_offset;
        } else {
            slice[addr.byte_offset] &= !(1 << addr.bit_offset);
        }
        Ok(())
    }

    /// Reads a 16-bit unsigned integer (%IW, %QW, %MW).
    pub fn read_u16(&self, addr: PlcAddress) -> ZeroResult<u16> {
        let slice = self.slice_for_area(addr.area);
        if addr.byte_offset + 2 > slice.len() {
            return Err(ZeroError::InvalidArgument);
        }
        Ok(u16::from_le_bytes([
            slice[addr.byte_offset],
            slice[addr.byte_offset + 1],
        ]))
    }

    /// Writes a 16-bit unsigned integer (%IW, %QW, %MW).
    pub fn write_u16(&mut self, addr: PlcAddress, val: u16) -> ZeroResult<()> {
        let slice = self.slice_for_area_mut(addr.area);
        if addr.byte_offset + 2 > slice.len() {
            return Err(ZeroError::InvalidArgument);
        }
        let bytes = val.to_le_bytes();
        slice[addr.byte_offset] = bytes[0];
        slice[addr.byte_offset + 1] = bytes[1];
        Ok(())
    }

    /// Reads a 32-bit IEEE 754 float (%ID, %QD, %MD).
    pub fn read_f32(&self, addr: PlcAddress) -> ZeroResult<f32> {
        let slice = self.slice_for_area(addr.area);
        if addr.byte_offset + 4 > slice.len() {
            return Err(ZeroError::InvalidArgument);
        }
        Ok(f32::from_le_bytes([
            slice[addr.byte_offset],
            slice[addr.byte_offset + 1],
            slice[addr.byte_offset + 2],
            slice[addr.byte_offset + 3],
        ]))
    }

    /// Writes a 32-bit IEEE 754 float (%ID, %QD, %MD).
    pub fn write_f32(&mut self, addr: PlcAddress, val: f32) -> ZeroResult<()> {
        let slice = self.slice_for_area_mut(addr.area);
        if addr.byte_offset + 4 > slice.len() {
            return Err(ZeroError::InvalidArgument);
        }
        let bytes = val.to_le_bytes();
        slice[addr.byte_offset..addr.byte_offset + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Copies external hardware input values into `%I` process input image.
    pub fn update_inputs(&mut self, raw: &[u8]) {
        let copy_len = raw.len().min(IN_BYTES);
        self.inputs[..copy_len].copy_from_slice(&raw[..copy_len]);
    }

    /// Copies `%Q` process output image to external hardware buffer.
    pub fn copy_outputs(&self, dest: &mut [u8]) {
        let copy_len = dest.len().min(OUT_BYTES);
        dest[..copy_len].copy_from_slice(&self.outputs[..copy_len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plc_bit_addressing() {
        let mut mem = PlcMemory::<64, 64, 256>::new();
        let addr = PlcAddress::bit(PlcArea::Outputs, 0, 3); // %QX0.3

        assert!(!mem.read_bit(addr).unwrap());
        mem.write_bit(addr, true).unwrap();
        assert!(mem.read_bit(addr).unwrap());
        assert_eq!(mem.outputs[0], 0x08);

        mem.write_bit(addr, false).unwrap();
        assert!(!mem.read_bit(addr).unwrap());
        assert_eq!(mem.outputs[0], 0x00);
    }

    #[test]
    fn test_plc_f32_rw() {
        let mut mem = PlcMemory::<64, 64, 256>::new();
        let addr = PlcAddress::dword(PlcArea::Markers, 16); // %MD16

        mem.write_f32(addr, 123.456).unwrap();
        let read = mem.read_f32(addr).unwrap();
        assert!((read - 123.456).abs() < 1e-4);
    }
}
