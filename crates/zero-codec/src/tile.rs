//! ZeroTile tile-based compression primitives and headers.

/// Pixel dimension of a square tile ($32 \times 32$).
pub const TILE_DIM: usize = 32;

/// Total pixels per tile ($32 \times 32 = 1024$).
pub const TILE_PIXELS: usize = TILE_DIM * TILE_DIM;

/// Uncompressed 32bpp byte size of a single tile ($1024 \times 4 = 4096$ bytes, matching 1 OS page).
pub const TILE_RAW_BYTES: usize = TILE_PIXELS * 4;

/// Category of compressed tile payload.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileKind {
    /// Zero payload: Tile is identical to reference frame (0 bytes).
    Unchanged = 0x00,
    /// Solid single color: 4 bytes payload `[B, G, R, A]`.
    SolidColor = 0x01,
    /// Raw uncompressed 4096-byte XOR difference.
    DeltaRaw = 0x02,
    /// Run-Length compressed XOR difference.
    DeltaRle = 0x03,
    /// Full keyframe intra tile (Run-Length compressed).
    FullIntra = 0x04,
}

impl TileKind {
    /// Decodes tile kind from wire byte.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::Unchanged),
            0x01 => Some(Self::SolidColor),
            0x02 => Some(Self::DeltaRaw),
            0x03 => Some(Self::DeltaRle),
            0x04 => Some(Self::FullIntra),
            _ => None,
        }
    }
}

/// Binary wire header accompanying every encoded tile.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileHeader {
    /// Horizontal tile coordinate index.
    pub tile_x: u16,
    /// Vertical tile coordinate index.
    pub tile_y: u16,
    /// Kind of tile payload.
    pub kind: u8,
    /// Bit flags (e.g. 0x01 = intra refresh).
    pub flags: u8,
    /// Length of payload in bytes following this header.
    pub payload_len: u16,
}

impl TileHeader {
    /// Byte length of binary tile header (8 bytes).
    pub const SIZE: usize = 8;

    /// Serializes tile header into a byte array.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut b = [0u8; Self::SIZE];
        b[0..2].copy_from_slice(&self.tile_x.to_le_bytes());
        b[2..4].copy_from_slice(&self.tile_y.to_le_bytes());
        b[4] = self.kind;
        b[5] = self.flags;
        b[6..8].copy_from_slice(&self.payload_len.to_le_bytes());
        b
    }

    /// Deserializes tile header from a byte slice.
    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < Self::SIZE {
            return None;
        }
        let tile_x = u16::from_le_bytes([slice[0], slice[1]]);
        let tile_y = u16::from_le_bytes([slice[2], slice[3]]);
        let kind = slice[4];
        let flags = slice[5];
        let payload_len = u16::from_le_bytes([slice[6], slice[7]]);

        Some(Self {
            tile_x,
            tile_y,
            kind,
            flags,
            payload_len,
        })
    }
}

/// Encoded tile with metadata header and compressed payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroTile {
    /// Tile header.
    pub header: TileHeader,
    /// Compressed payload bytes.
    pub payload: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_header_serialization() {
        let h = TileHeader {
            tile_x: 12,
            tile_y: 34,
            kind: TileKind::DeltaRle as u8,
            flags: 0x01,
            payload_len: 128,
        };
        let b = h.to_bytes();
        let parsed = TileHeader::from_bytes(&b).expect("Parse header");
        assert_eq!(parsed.tile_x, 12);
        assert_eq!(parsed.tile_y, 34);
        assert_eq!(parsed.kind, TileKind::DeltaRle as u8);
        assert_eq!(parsed.flags, 0x01);
        assert_eq!(parsed.payload_len, 128);
    }
}
